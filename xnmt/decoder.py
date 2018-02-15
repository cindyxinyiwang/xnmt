import dynet as dy
import numpy as np
from xnmt.serializer import Serializable
import xnmt.batcher
from xnmt.events import register_handler, handle_xnmt_event
import xnmt.linear
from input import *
from lstm import CustomCompactLSTMBuilder

class Decoder(object):
  '''
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  '''

  '''
  Document me
  '''

  def calc_loss(self, x, ref_action):
    raise NotImplementedError('calc_loss must be implemented in Decoder subclasses')

class RnnDecoder(Decoder):
  @staticmethod
  def rnn_from_spec(spec, num_layers, input_dim, hidden_dim, model, residual_to_output):
    decoder_type = spec.lower()
    if decoder_type == "lstm":
      return dy.CompactVanillaLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    elif decoder_type == "residuallstm":
      return residual.ResidualRNNBuilder(num_layers, input_dim, hidden_dim,
                                         model, residual_to_output)
    elif decoder_type == "custom":
      return CustomCompactLSTMBuilder(num_layers, input_dim, hidden_dim, model)
    else:
      raise RuntimeError("Unknown decoder type {}".format(spec))

class MlpSoftmaxDecoderState(object):
  """A state holding all the information needed for MLPSoftmaxDecoder"""
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!MlpSoftmaxDecoder'

  def __init__(self, yaml_context, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None):
    register_handler(self)
    param_col = yaml_context.dynet_param_collection.param_col
    # Define dim
    lstm_dim       = lstm_dim or yaml_context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or yaml_context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or yaml_context.default_layer_dim
    input_dim      = input_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(yaml_context, self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)
    # MLP
    self.context_projector = xnmt.linear.Linear(input_dim  = input_dim + lstm_dim,
                                           output_dim = mlp_hidden_dim,
                                           model = param_col)
    self.vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = param_col)
    # Dropout
    self.dropout = dropout or yaml_context.dropout

  def shared_params(self):
    return [set(["layers", "bridge.dec_layers"]),
            set(["lstm_dim", "bridge.dec_dim"])]

  def initial_state(self, enc_final_states, ss_expr):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    rnn_state = self.fwd_lstm.initial_state()
    rnn_state = rnn_state.set_s(self.bridge.decoder_init(enc_final_states))
    zeros = dy.zeros(self.input_dim) if self.input_feeding else None
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    return MlpSoftmaxDecoderState(rnn_state=rnn_state, context=zeros)

  def add_input(self, mlp_dec_state, trg_embedding):
    """Add an input and update the state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :returns: The update MLP decoder state.
    """
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, mlp_dec_state.context])
    return MlpSoftmaxDecoderState(rnn_state=mlp_dec_state.rnn_state.add_input(inp),
                                  context=mlp_dec_state.context)

  def get_scores(self, mlp_dec_state):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    h_t = dy.tanh(self.context_projector(dy.concatenate([mlp_dec_state.rnn_state.output(), mlp_dec_state.context])))
    return self.vocab_projector(h_t)

  def calc_loss(self, mlp_dec_state, ref_action):
    scores = self.get_scores(mlp_dec_state)
    # single mode
    if not xnmt.batcher.is_batched(ref_action):
      return dy.pickneglogsoftmax(scores, ref_action)
    # minibatch mode
    else:
      return dy.pickneglogsoftmax_batch(scores, ref_action)

  @handle_xnmt_event
  def on_set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)

class OpenNonterm:
  def __init__(self, label=None, parent_state=None, is_sibling=False, sib_state=None):
    self.label = label 
    self.parent_state = parent_state
    self.is_sibling = is_sibling
    self.sib_state = sib_state

class TreeDecoderState:
  """A state holding all the information needed for MLPSoftmaxDecoder"""
  def __init__(self, rnn_state=None, word_rnn_state=None, context=None, word_context=None,
               states=[], tree=None, open_nonterms=[], prev_word_state=None, stop_action=False, leaf_len=-1, step_len=-1):
    self.rnn_state = rnn_state
    self.context = context
    self.word_rnn_state = word_rnn_state
    self.word_context = word_context
    # training time
    self.states = states
    self.tree = tree

    #used at decoding time
    self.open_nonterms = open_nonterms
    self.prev_word_state = prev_word_state
    self.stop_action = stop_action
    self.leaf_len = leaf_len
    self.step_len = step_len

  def copy(self):
    open_nonterms_copy = []
    for n in self.open_nonterms:
      open_nonterms_copy.append(OpenNonterm(n.label, n.parent_state, n.is_sibling, n.sib_state))
    return TreeDecoderState(rnn_state=self.rnn_state, word_rnn_state=self.word_rnn_state, context=self.context, word_context=self.word_context,
                            open_nonterms=open_nonterms_copy, prev_word_state=self.prev_word_state, stop_action=self.stop_action,
                            leaf_len=self.leaf_len, step_len=self.step_len)

class TreeDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!TreeDecoder'


  def __init__(self, yaml_context, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, tag_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True, word_state_feeding=True,
               bridge=None, word_lstm=False, start_nonterm='ROOT', frontir_feeding=False):
    register_handler(self)
    param_col = yaml_context.dynet_param_collection.param_col
    self.start_nonterm = start_nonterm
    # Define dim
    lstm_dim       = lstm_dim or yaml_context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or yaml_context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or yaml_context.default_layer_dim
    tag_embed_dim = tag_embed_dim or yaml_context.default_layer_dim
    input_dim      = input_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    self.trg_embed_dim = trg_embed_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    plain_lstm_input = lstm_input
    # parent state + last_word_state + frontir embedding
    lstm_input += lstm_dim
    self.word_state_feeding = word_state_feeding
    if self.word_state_feeding:
      lstm_input += lstm_dim
    self.frontir_feeding = frontir_feeding
    if frontir_feeding:
      lstm_input += tag_embed_dim

    if word_lstm:
      lstm_input += lstm_dim
    self.init_lstm_dim = lstm_input - trg_embed_dim

    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(yaml_context, self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)

    self.set_word_lstm = word_lstm
    if word_lstm:
      self.word_lstm = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                                num_layers = layers,
                                                input_dim  = plain_lstm_input,
                                                hidden_dim = lstm_dim,
                                                model = param_col,
                                                residual_to_output = residual_to_output)
    # MLP
    self.context_projector = xnmt.linear.Linear(input_dim=input_dim + lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model=param_col)

    self.vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = param_col)
    # Dropout
    self.dropout = dropout or yaml_context.dropout

  def shared_params(self):
    return [set(["layers", "bridge.dec_layers"]),
            set(["lstm_dim", "bridge.dec_dim"])]

  def initial_state(self, enc_final_states, ss_expr, decoding=False):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    init_state = self.bridge.decoder_init(enc_final_states)
    if self.set_word_lstm:
      # init_state: [c, h]
      word_rnn_state = self.word_lstm.initial_state()
      word_rnn_state = word_rnn_state.set_s(init_state)
      zeros = dy.zeros(self.input_dim) if self.input_feeding else None
      word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    else:
      word_rnn_state = None

    rnn_state = self.fwd_lstm.initial_state()
    rnn_state = rnn_state.set_s(init_state)

    zeros = dy.zeros(self.init_lstm_dim)
    #if self.set_word_lstm:
    #  zeros = dy.concatenate([zeros, word_rnn_state.output()])
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))

    self.decoding = decoding
    if decoding:
      zeros_lstm = dy.zeros(self.lstm_dim)
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, word_context=zeros,  \
          open_nonterms=[OpenNonterm(self.start_nonterm, parent_state=zeros_lstm, sib_state=zeros_lstm)], \
          prev_word_state=zeros_lstm)
    else:     
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, word_context=zeros, \
          states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]))

  def add_input(self, tree_dec_state, trg_embedding, trg, trg_rule_vocab=None, tag_embedder=None):
    """Add an input and update the state.

    :param tree_dec_state: An TreeDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :param trg: The data list of the target word, with the first element as the word index, the rest as timestep.
    :param trg_rule_vocab: RuleVocab object used at decoding time
    :returns: The update MLP decoder state.
    """
    inp = trg_embedding
    if self.input_feeding:
      inp = dy.concatenate([inp, tree_dec_state.context])
    if not self.decoding:
      # get parent states for this batch
      batch_size = trg_embedding.dim()[1]
      assert batch_size == 1

      paren_tm1_states = tree_dec_state.states[trg.get_col(1)] # ((hidden_dim,), batch_size) * batch_size
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_state = paren_tm1_states[0]

      if self.word_state_feeding:
        last_word_states = tree_dec_state.states[trg.get_col(2)]
        last_word_state = last_word_states[0]
        inp = dy.concatenate([inp, paren_tm1_state, last_word_state])
      else:
        inp = dy.concatenate([inp, paren_tm1_state])
      if self.frontir_feeding:
        frontir_list = trg.get_col(5, batched=False)
        frontir_emb = tag_embedder.embed(frontir_list[0])
        inp = dy.concatenate([inp, frontir_emb])
      # get word_rnn state
      word_rnn_state = tree_dec_state.word_rnn_state
      if self.set_word_lstm:
        if is_terminal[0] == 1:
          word_inp = trg_embedding
          if self.input_feeding:
            word_inp = dy.concatenate([word_inp, tree_dec_state.word_context])
          word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp)
        inp = dy.concatenate([inp, word_rnn_state.output()])

      rnn_state = tree_dec_state.rnn_state.add_input(inp)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context, \
                           states=np.append(tree_dec_state.states, rnn_state.output()))
    else:
      cur_nonterm = tree_dec_state.open_nonterms.pop()
      rule = trg_rule_vocab[trg]
      if cur_nonterm.label != rule.lhs:
        for c in cur_nonterm:
          print c.label
      assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)
      # find frontier node label
      if self.word_state_feeding:
        inp = dy.concatenate([inp, cur_nonterm.parent_state, tree_dec_state.prev_word_state])
      else:
        inp = dy.concatenate([inp, cur_nonterm.parent_state])
      if self.frontir_feeding:
        frontir_emb = tag_embedder.embed(trg_rule_vocab.tag_vocab.convert(rule.lhs))
        inp = dy.concatenate([inp, frontir_emb])

      word_rnn_state = tree_dec_state.word_rnn_state
      if self.set_word_lstm:
        if len(rule.open_nonterms) == 0:
          word_inp = trg_embedding
          if self.input_feeding:
            word_inp = dy.concatenate([word_inp, tree_dec_state.word_context])
          word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp)
        inp = dy.concatenate([inp, word_rnn_state.output()])

      rnn_state = tree_dec_state.rnn_state.add_input(inp)
      # add rule to tree_dec_state.open_nonterms
      prev_word_state = tree_dec_state.prev_word_state
      open_nonterms=tree_dec_state.open_nonterms[:]
      new_open_nonterms = []
      for rhs in rule.rhs:
        if rhs in rule.open_nonterms:
          new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output()))
        else:
          prev_word_state = rnn_state.output()
      new_open_nonterms.reverse()
      open_nonterms.extend(new_open_nonterms)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context,\
                              open_nonterms=open_nonterms, prev_word_state=prev_word_state)

  def get_scores(self, tree_dec_state, trg_rule_vocab, label_idx=-1, is_terminal=None):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    #if self.set_word_lstm:
      #h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context])))
      #h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context,
      #                                                     tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context])))
    #else:
    h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context])))
    if label_idx >= 0:
      return self.vocab_projector(h_t), 0
      label = trg_rule_vocab.tag_vocab[label_idx]
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(label)
    else:
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.open_nonterms[-1].label)
    if not valid_y_index:
      valid_y_index = [i for i in range(len(trg_rule_vocab))]
    valid_y_mask = np.ones((len(trg_rule_vocab),)) * (-1000)
    valid_y_mask[valid_y_index] = 0.
    return self.vocab_projector(h_t) + dy.inputTensor(valid_y_mask), len(valid_y_index)

  def calc_loss(self, tree_dec_state, ref_action, trg_rule_vocab):
    ref_word = ref_action.get_col(0)
    scores, valid_y_len = self.get_scores(tree_dec_state, trg_rule_vocab, ref_action.get_col(5)[0])
    # single mode
    if not xnmt.batcher.is_batched(ref_action):
      return dy.pickneglogsoftmax(scores, ref_word)
    # minibatch mode
    else:
      return dy.pickneglogsoftmax_batch(scores, ref_word)

  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)
    if self.set_word_lstm:
      self.word_lstm.set_dropout(self.dropout if val else 0.0)

class TreeHierFixtransDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!TreeHierFixtransDecoder'


  def __init__(self, yaml_context, vocab_size, word_vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, tag_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None, start_nonterm='ROOT'):

    register_handler(self)
    self.set_word_lstm = True
    param_col = yaml_context.dynet_param_collection.param_col
    self.start_nonterm = start_nonterm

    # Define dim
    lstm_dim       = lstm_dim or yaml_context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or yaml_context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or yaml_context.default_layer_dim
    tag_embed_dim = tag_embed_dim or yaml_context.default_layer_dim
    input_dim      = input_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    self.trg_embed_dim = trg_embed_dim
    rule_lstm_input = trg_embed_dim
    word_lstm_input = trg_embed_dim
    if input_feeding:
      rule_lstm_input += input_dim
      word_lstm_input += input_dim

    # parent state + wordRNN output
    rule_lstm_input += lstm_dim*2
    # ruleRNN output
    word_lstm_input += lstm_dim

    self.rule_lstm_input = rule_lstm_input
    self.word_lstm_input = word_lstm_input
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(yaml_context, self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = rule_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)

    self.word_lstm = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = word_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)
    # MLP
    self.rule_context_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model=param_col)
    self.word_context_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model=param_col)
    self.rule_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = param_col)
    self.word_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = word_vocab_size,
                                         model = param_col)

    # Dropout
    self.dropout = dropout or yaml_context.dropout

  def shared_params(self):
    return [set(["layers", "bridge.dec_layers"]),
            set(["lstm_dim", "bridge.dec_dim"])]

  def initial_state(self, enc_final_states, ss_expr, decoding=False):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    init_state = self.bridge.decoder_init(enc_final_states)
    # init_state: [c, h]
    word_rnn_state = self.word_lstm.initial_state()
    word_rnn_state = word_rnn_state.set_s(init_state)
    zeros_word_rnn = dy.zeros(self.word_lstm_input - self.trg_embed_dim)
    word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros_word_rnn]))

    rnn_state = self.fwd_lstm.initial_state()
    rnn_state = rnn_state.set_s(init_state)
    zeros_rnn = dy.zeros(self.rule_lstm_input - self.trg_embed_dim)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros_rnn]))

    self.decoding = decoding
    if decoding:
      zeros_lstm = dy.zeros(self.lstm_dim)
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn,  \
          open_nonterms=[OpenNonterm(self.start_nonterm, parent_state=zeros_lstm, sib_state=zeros_lstm)], \
          prev_word_state=zeros_lstm)
    else:
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn, \
          states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]))

  def add_input(self, tree_dec_state, trg, word_embedder, rule_embedder,
                trg_rule_vocab=None, word_vocab=None, tag_embedder=None):
    """Add an input and update the state.

    :param tree_dec_state: An TreeDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :param trg: The data list of the target word, with the first element as the word index, the rest as timestep.
    :param trg_rule_vocab: RuleVocab object used at decoding time
    :returns: The update MLP decoder state.
    """
    word_rnn_state = tree_dec_state.word_rnn_state
    rnn_state = tree_dec_state.rnn_state
    if not self.decoding:
      # get parent states for this batch
      #batch_size = trg_embedding.dim()[1]
      #assert batch_size == 1
      states = tree_dec_state.states
      paren_tm1_states = tree_dec_state.states[trg.get_col(1)] # ((hidden_dim,), batch_size) * batch_size
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_state = paren_tm1_states[0]
      if is_terminal[0] == 0:
        # rule rnn
        rule_idx = trg.get_col(0)
        #print trg_rule_vocab[trg.get_col(0, batched=False)[0]]
        inp = rule_embedder.embed(rule_idx)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])
        inp = dy.concatenate([inp, paren_tm1_state, word_rnn_state.output()])

        rnn_state = rnn_state.add_input(inp)
        states = np.append(states, rnn_state.output())
      else:
        # word rnn
        word_idx = trg.get_col(0)
        # if this is end of phrase append states list
        #print word_vocab[trg.get_col(0, batched=False)[0]].encode('utf-8')
        if word_idx[0] == Vocab.ES:
          states = np.append(states, rnn_state.output())
        else:
          inp = word_embedder.embed(word_idx)
          emb = inp
          if self.input_feeding:
            inp = dy.concatenate([inp, tree_dec_state.word_context])
          inp = dy.concatenate([inp, paren_tm1_state])
          word_rnn_state = word_rnn_state.add_input(inp)
          # update rule RNN
          rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                        word_rnn_state.output()])
          rnn_state = rnn_state.add_input(rnn_inp)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context, \
                           states=states)
    else:
      #print trg
      open_nonterms = tree_dec_state.open_nonterms[:]
      prev_word_state = tree_dec_state.prev_word_state
      if open_nonterms[-1].label == u'*':
        if trg == Vocab.ES:
          open_nonterms.pop()
        else:
          inp = word_embedder.embed(trg)
          emb = inp
          if self.input_feeding:
            inp = dy.concatenate([inp, tree_dec_state.word_context])
          inp = dy.concatenate([inp, tree_dec_state.open_nonterms[-1].parent_state])
          word_rnn_state = word_rnn_state.add_input(inp)

          rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                     word_rnn_state.output()])
          rnn_state = rnn_state.add_input(rnn_inp)
      else:
        inp = rule_embedder.embed(trg)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])

        cur_nonterm = open_nonterms.pop()
        rule = trg_rule_vocab[trg]
        if cur_nonterm.label != rule.lhs:
          for c in cur_nonterm:
            print c.label
        assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)

        inp = dy.concatenate([inp, cur_nonterm.parent_state, word_rnn_state.output()])
        rnn_state = rnn_state.add_input(inp)
        # add rule to tree_dec_state.open_nonterms
        new_open_nonterms = []
        for rhs in rule.rhs:
          if rhs in rule.open_nonterms:
            new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output()))
          else:
            prev_word_state = rnn_state.output()
        new_open_nonterms.reverse()
        open_nonterms.extend(new_open_nonterms)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context,\
                              open_nonterms=open_nonterms, prev_word_state=prev_word_state)

  def init_wordRNN(self, tree_dec_state, prev_word_emb=None, paren_t=None):
    word_rnn_state = tree_dec_state.word_rnn_state
    rnn_state = tree_dec_state.rnn_state
    if self.decoding:
      #inp = tree_dec_state.prev_word_emb
      inp = prev_word_emb
      if self.input_feeding:
        inp = dy.concatenate([inp, tree_dec_state.word_context])
      # inp = dy.concatenate([inp, tree_dec_state.open_nonterms[-1].parent_state, dy.zeros(self.lstm_dim)])
      inp = dy.concatenate([inp, tree_dec_state.open_nonterms[-1].parent_state])
      word_rnn_state = word_rnn_state.add_input(inp)
    else:
      inp = prev_word_emb
      if self.input_feeding:
        inp = dy.concatenate([inp, tree_dec_state.word_context])
      inp = dy.concatenate([inp, tree_dec_state.states[paren_t][0]])
      word_rnn_state = word_rnn_state.add_input(inp)
    rnn_state = rnn_state.add_input(dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                                    word_rnn_state.output()]))
    tree_dec_state.word_rnn_state = word_rnn_state
    tree_dec_state.rnn_state = rnn_state
    return tree_dec_state

  def get_scores(self, tree_dec_state, trg_rule_vocab, is_terminal, label_idx=-1, sample_len=None):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    if is_terminal:
      inp = dy.concatenate([tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context,
                                                                tree_dec_state.rnn_state.output(), tree_dec_state.context])
      h_t = dy.tanh(self.word_context_projector(inp))
      return self.word_vocab_projector(h_t), -1, None, None
    else:
      inp = dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context,
                                                              tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context])
      h_t = dy.tanh(self.rule_context_projector(inp))
    if label_idx >= 0:
      # training
      return self.rule_vocab_projector(h_t), -1, None, None
      #label = trg_rule_vocab.tag_vocab[label_idx]
      #valid_y_index = trg_rule_vocab.rule_index_with_lhs(label)
    else:
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.open_nonterms[-1].label)
    if not valid_y_index:
      print 'warning: no rule with lhs: {}'.format(tree_dec_state.open_nonterms[-1].label)
      #valid_y_index = [i for i in range(len(trg_rule_vocab))]
    valid_y_mask = np.ones((len(trg_rule_vocab),)) * (-1000)
    valid_y_mask[valid_y_index] = 0.

    return self.rule_vocab_projector(h_t) + dy.inputTensor(valid_y_mask), len(valid_y_index), None, None

  def calc_loss(self, tree_dec_state, ref_action, trg_rule_vocab):
    ref_word = ref_action.get_col(0)
    is_terminal = ref_action.get_col(3)[0]
    is_stop =ref_action.get_col(4)[0]
    leaf_len = ref_action.get_col(5)[0]

    scores, valid_y_len, stop_prob, len_scores = self.get_scores(tree_dec_state, trg_rule_vocab,
                                                                 is_terminal, label_idx=1)
    # single mode
    if not xnmt.batcher.is_batched(ref_action):
      word_loss = dy.pickneglogsoftmax(scores, ref_word)
    # minibatch mode
    else:
      word_loss = dy.pickneglogsoftmax_batch(scores, ref_word)
    return word_loss

  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)
    if self.set_word_lstm:
      self.word_lstm.set_dropout(self.dropout if val else 0.0)

class TreeHierDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!TreeHierDecoder'


  def __init__(self, yaml_context, vocab_size, word_vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, tag_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None, start_nonterm='ROOT', feed_word_emb=False, action_loss_weight=-1, len_loss_weight=-1, rule_cond=True,
               update_ruleRNN=True, rule_label_smooth=-1):

    register_handler(self)
    self.set_word_lstm = True
    param_col = yaml_context.dynet_param_collection.param_col
    self.start_nonterm = start_nonterm
    self.feed_word_emb = feed_word_emb
    self.update_ruleRNN = update_ruleRNN
    self.rule_label_smooth = rule_label_smooth
    self.rule_size = vocab_size

    if action_loss_weight > 0:
      self.action_loss = True
      self.action_loss_weight = action_loss_weight
    else:
      self.action_loss = False
    if len_loss_weight > 0:
      self.len_loss = True
      self.len_loss_weight = len_loss_weight
    else:
      self.len_loss = False
    # Define dim
    lstm_dim       = lstm_dim or yaml_context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or yaml_context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or yaml_context.default_layer_dim
    tag_embed_dim = tag_embed_dim or yaml_context.default_layer_dim
    input_dim      = input_dim or yaml_context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    self.trg_embed_dim = trg_embed_dim
    rule_lstm_input = trg_embed_dim
    word_lstm_input = trg_embed_dim
    if input_feeding:
      rule_lstm_input += input_dim
      word_lstm_input += input_dim

    # parent state + wordRNN output
    rule_lstm_input += lstm_dim*2
    if self.feed_word_emb:
      rule_lstm_input += trg_embed_dim
    # ruleRNN output
    word_lstm_input += lstm_dim

    self.rule_lstm_input = rule_lstm_input
    self.word_lstm_input = word_lstm_input
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(yaml_context, self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = rule_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)

    self.word_lstm = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = word_lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)
    # MLP
    self.rule_cond = rule_cond
    if rule_cond:
      rule_input = 2*input_dim + 2*lstm_dim
    else:
      rule_input = input_dim + lstm_dim
    self.rule_context_projector = xnmt.linear.Linear(input_dim=rule_input,
                                                output_dim=mlp_hidden_dim,
                                                model=param_col)
    self.word_context_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                output_dim=mlp_hidden_dim,
                                                model=param_col)
    self.rule_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = vocab_size,
                                         model = param_col)
    self.word_vocab_projector = xnmt.linear.Linear(input_dim = mlp_hidden_dim,
                                         output_dim = word_vocab_size,
                                         model = param_col)
    if self.action_loss:
      #self.action_context_projector = xnmt.linear.Linear(input_dim=2*lstm_dim+trg_embed_dim,
      #                                                output_dim=mlp_hidden_dim,
      #                                                model=param_col)
      self.action_projector = xnmt.linear.Linear(input_dim=2*input_dim + 2*lstm_dim,
                                                 output_dim=1,
                                                 model=param_col)
    if self.len_loss:
      self.len_context_projector = xnmt.linear.Linear(input_dim=2*lstm_dim+2*input_dim,
                                                      output_dim=mlp_hidden_dim,
                                                      model=param_col)
      self.len_projector = xnmt.linear.Linear(input_dim=mlp_hidden_dim,
                                              output_dim=25,
                                              model=param_col)
    # Dropout
    self.dropout = dropout or yaml_context.dropout

  def shared_params(self):
    return [set(["layers", "bridge.dec_layers"]),
            set(["lstm_dim", "bridge.dec_dim"])]

  def initial_state(self, enc_final_states, ss_expr, decoding=False):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    init_state = self.bridge.decoder_init(enc_final_states)
    # init_state: [c, h]
    word_rnn_state = self.word_lstm.initial_state()
    word_rnn_state = word_rnn_state.set_s(init_state)
    zeros_word_rnn = dy.zeros(self.word_lstm_input - self.trg_embed_dim)
    word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros_word_rnn]))

    rnn_state = self.fwd_lstm.initial_state()
    rnn_state = rnn_state.set_s(init_state)
    zeros_rnn = dy.zeros(self.rule_lstm_input - self.trg_embed_dim)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros_rnn]))

    self.decoding = decoding
    if decoding:
      zeros_lstm = dy.zeros(self.lstm_dim)
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn,  \
          open_nonterms=[OpenNonterm(self.start_nonterm, parent_state=zeros_lstm, sib_state=zeros_lstm)], \
          prev_word_state=zeros_lstm)
    else:
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros_rnn, word_rnn_state=word_rnn_state, word_context=zeros_word_rnn, \
          states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]))

  def add_input(self, tree_dec_state, trg, word_embedder, rule_embedder,
                trg_rule_vocab=None, word_vocab=None, tag_embedder=None):
    """Add an input and update the state.

    :param tree_dec_state: An TreeDecoderState object containing the current state.
    :param trg_embedding: The embedding of the word to input.
    :param trg: The data list of the target word, with the first element as the word index, the rest as timestep.
    :param trg_rule_vocab: RuleVocab object used at decoding time
    :returns: The update MLP decoder state.
    """
    word_rnn_state = tree_dec_state.word_rnn_state
    rnn_state = tree_dec_state.rnn_state
    if not self.decoding:
      # get parent states for this batch
      #batch_size = trg_embedding.dim()[1]
      #assert batch_size == 1
      states = tree_dec_state.states
      paren_tm1_states = tree_dec_state.states[trg.get_col(1)] # ((hidden_dim,), batch_size) * batch_size
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_state = paren_tm1_states[0]
      if is_terminal[0] == 0:
        # rule rnn
        rule_idx = trg.get_col(0)
        #print trg_rule_vocab[trg.get_col(0, batched=False)[0]]
        inp = rule_embedder.embed(rule_idx)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])
        inp = dy.concatenate([inp, paren_tm1_state, word_rnn_state.output()])
        if self.feed_word_emb:
          inp = dy.concatenate([inp, word_embedder.embed(Vocab.ES)])
        rnn_state = rnn_state.add_input(inp)
        states = np.append(states, rnn_state.output())

        #word_rnn_state = word_rnn_state.add_input(dy.concatenate([dy.zeros(self.word_lstm_input-self.lstm_dim),
        #                                                          rnn_state.output()]))
      else:
        # word rnn
        word_idx = trg.get_col(0)
        # if this is end of phrase append states list
        #print word_vocab[trg.get_col(0, batched=False)[0]].encode('utf-8')
        inp = word_embedder.embed(word_idx)
        emb = inp
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.word_context])
        inp = dy.concatenate([inp, paren_tm1_state])
        word_rnn_state = word_rnn_state.add_input(inp)
        # update rule RNN
        if self.update_ruleRNN:
          if self.feed_word_emb:
            rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim-self.trg_embed_dim),
                                      word_rnn_state.output(), emb])
          else:
            rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                      word_rnn_state.output()])
          rnn_state = rnn_state.add_input(rnn_inp)
        # if this is end of phrase append states list
        if self.action_loss or self.len_loss:
          action = trg.get_col(4)[0]
          if action == 1:
            states = np.append(states, rnn_state.output())
        else:
          if word_idx[0] == Vocab.ES:
            states = np.append(states, rnn_state.output())

      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context, \
                           states=states)
    else:
      #print trg
      open_nonterms = tree_dec_state.open_nonterms[:]
      prev_word_state = tree_dec_state.prev_word_state
      leaf_len = tree_dec_state.leaf_len
      step_len = tree_dec_state.step_len
      if open_nonterms[-1].label == u'*':
        inp = word_embedder.embed(trg)
        emb = inp
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.word_context])
        inp = dy.concatenate([inp, tree_dec_state.open_nonterms[-1].parent_state])
        word_rnn_state = word_rnn_state.add_input(inp)

        if self.update_ruleRNN:
          if self.feed_word_emb:
            rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim-self.trg_embed_dim),
                                      word_rnn_state.output(), emb])
          else:
            rnn_inp = dy.concatenate([dy.zeros(self.rule_lstm_input - self.lstm_dim),
                                      word_rnn_state.output()])
          rnn_state = rnn_state.add_input(rnn_inp)

        if self.action_loss:
          if tree_dec_state.stop_action:
            open_nonterms.pop()
        elif self.len_loss:
          step_len += 1
          if leaf_len == step_len:
            open_nonterms.pop()
            leaf_len = -1
            step_len = -1
        else:
          if trg == Vocab.ES:
            open_nonterms.pop()

      else:
        inp = rule_embedder.embed(trg)
        if self.input_feeding:
          inp = dy.concatenate([inp, tree_dec_state.context])

        cur_nonterm = open_nonterms.pop()
        rule = trg_rule_vocab[trg]
        if cur_nonterm.label != rule.lhs:
          for c in cur_nonterm:
            print c.label
        assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)

        inp = dy.concatenate([inp, cur_nonterm.parent_state, word_rnn_state.output()])
        if self.feed_word_emb:
          inp = dy.concatenate([inp, word_embedder.embed(Vocab.ES)])
        rnn_state = rnn_state.add_input(inp)
        # add rule to tree_dec_state.open_nonterms
        new_open_nonterms = []
        for rhs in rule.rhs:
          if rhs in rule.open_nonterms:
            new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output()))
          else:
            prev_word_state = rnn_state.output()
        new_open_nonterms.reverse()
        open_nonterms.extend(new_open_nonterms)

        #word_rnn_state = word_rnn_state.add_input(dy.concatenate([dy.zeros(self.word_lstm_input - self.lstm_dim),
        #                                                          rnn_state.output()]))
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, word_context=tree_dec_state.word_context,\
                              open_nonterms=open_nonterms, prev_word_state=prev_word_state, leaf_len=leaf_len, step_len=step_len)

  def get_scores(self, tree_dec_state, trg_rule_vocab, is_terminal, label_idx=-1, sample_len=False):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    if is_terminal:
      inp = dy.concatenate([tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context,
                                                                tree_dec_state.rnn_state.output(), tree_dec_state.context])
      h_t = dy.tanh(self.word_context_projector(inp))
      if self.action_loss:
        stop_prob = dy.logistic(self.action_projector(inp))
        return self.word_vocab_projector(h_t), -1, stop_prob, None
      elif self.len_loss and sample_len:
        h_len_t = dy.tanh(self.len_context_projector(inp))
        len_scores = self.len_projector(h_len_t)
        return self.word_vocab_projector(h_t), -1, None, len_scores
      else:
        return self.word_vocab_projector(h_t), -1, None, None
    else:
      if self.rule_cond:
        inp = dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context,
                                                                  tree_dec_state.word_rnn_state.output(), tree_dec_state.word_context])
      else:
        inp = dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context])
      h_t = dy.tanh(self.rule_context_projector(inp))
    if self.rule_label_smooth > 0:
      proj = dy.cmult(self.rule_vocab_projector(h_t), dy.scalarInput(1. - self.rule_label_smooth)) \
             + dy.scalarInput(self.rule_label_smooth / float(self.rule_size))
    else:
      proj = self.rule_vocab_projector(h_t)
    if label_idx >= 0:
      # training
      return proj, -1, None, None
      #label = trg_rule_vocab.tag_vocab[label_idx]
      #valid_y_index = trg_rule_vocab.rule_index_with_lhs(label)
    else:
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.open_nonterms[-1].label)
    if not valid_y_index:
      print 'warning: no rule with lhs: {}'.format(tree_dec_state.open_nonterms[-1].label)
    valid_y_mask = np.ones((len(trg_rule_vocab),)) * (-1000)
    valid_y_mask[valid_y_index] = 0.

    return proj + dy.inputTensor(valid_y_mask), len(valid_y_index), None, None

  def calc_loss(self, tree_dec_state, ref_action, trg_rule_vocab):
    ref_word = ref_action.get_col(0)
    is_terminal = ref_action.get_col(3)[0]
    is_stop =ref_action.get_col(4)[0]
    leaf_len = ref_action.get_col(5)[0]

    scores, valid_y_len, stop_prob, len_scores = self.get_scores(tree_dec_state, trg_rule_vocab,
                                                                 is_terminal, label_idx=1, sample_len=leaf_len>0)
    # single mode
    if not xnmt.batcher.is_batched(ref_action):
      word_loss = dy.pickneglogsoftmax(scores, ref_word)
    # minibatch mode
    else:
      word_loss = dy.pickneglogsoftmax_batch(scores, ref_word)
    if self.action_loss and is_terminal:
      if is_stop:
        word_loss = word_loss - dy.log(stop_prob) * self.action_loss_weight
      else:
        word_loss = word_loss - dy.log(1. - stop_prob) * self.action_loss_weight
      #print is_stop, stop_prob.value()
    if self.len_loss and is_terminal and leaf_len > 0:
      if leaf_len > 20: leaf_len = 20
      word_loss += dy.pickneglogsoftmax(len_scores, leaf_len-1) * self.len_loss_weight
    return word_loss

  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)
    if self.set_word_lstm:
      self.word_lstm.set_dropout(self.dropout if val else 0.0)

class Bridge(object):
  """
  Responsible for initializing the decoder LSTM, based on the final encoder state
  """
  def decoder_init(self, dec_layers, dec_dim, enc_final_states):
    raise NotImplementedError("decoder_init() must be implemented by Bridge subclasses")

class NoBridge(Bridge, Serializable):
  """
  This bridge initializes the decoder with zero vectors, disregarding the encoder final states.
  """
  yaml_tag = u'!NoBridge'
  def __init__(self, yaml_context, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or yaml_context.default_layer_dim
  def decoder_init(self, enc_final_states):
    batch_size = enc_final_states[0].main_expr().dim()[1]
    z = dy.zeros(self.dec_dim, batch_size)
    return [z] * (self.dec_layers * 2)

class CopyBridge(Bridge, Serializable):
  """
  This bridge copies final states from the encoder to the decoder initial states.
  Requires that:
  - encoder / decoder dimensions match for every layer
  - num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)
  """
  yaml_tag = u'!CopyBridge'
  def __init__(self, yaml_context, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or yaml_context.default_layer_dim
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("CopyBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.dec_dim:
      raise RuntimeError("CopyBridge requires enc_dim == dec_dim, but got %s and %s" % (enc_final_states[0].main_expr().dim()[0][0], self.dec_dim))
    return [enc_state.cell_expr() for enc_state in enc_final_states[-self.dec_layers:]] \
         + [enc_state.main_expr() for enc_state in enc_final_states[-self.dec_layers:]]

class LinearBridge(Bridge, Serializable):
  """
  This bridge does a linear transform of final states from the encoder to the decoder initial states.
  Requires that:
  - num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)
  """
  yaml_tag = u'!LinearBridge'
  def __init__(self, yaml_context, dec_layers, enc_dim = None, dec_dim = None):
    param_col = yaml_context.dynet_param_collection.param_col
    self.dec_layers = dec_layers
    self.enc_dim = enc_dim or yaml_context.default_layer_dim
    self.dec_dim = dec_dim or yaml_context.default_layer_dim
    self.projector = xnmt.linear.Linear(input_dim  = enc_dim,
                                           output_dim = dec_dim,
                                           model = param_col)
  def decoder_init(self, enc_final_states):
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("LinearBridge requires dec_layers <= len(enc_final_states), but got %s and %s" % (self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.enc_dim:
      raise RuntimeError("LinearBridge requires enc_dim == %s, but got %s" % (self.enc_dim, enc_final_states[0].main_expr().dim()[0][0]))
    decoder_init = [self.projector(enc_state.main_expr()) for enc_state in enc_final_states[-self.dec_layers:]]
    return decoder_init + [dy.tanh(dec) for dec in decoder_init]

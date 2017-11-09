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
  def __init__(self, rnn_state=None, word_rnn_state=None, context=None, states=[], tree=None, open_nonterms=[], prev_word_state=None):
    self.rnn_state = rnn_state
    self.word_rnn_state = word_rnn_state
    self.context = context
    # training time
    self.states = states
    self.tree = tree

    #used at decoding time
    self.open_nonterms = open_nonterms
    self.prev_word_state = prev_word_state

  def copy(self):
    open_nonterms_copy = []
    for n in self.open_nonterms:
      open_nonterms_copy.append(OpenNonterm(n.label, n.parent_state, n.is_sibling, n.sib_state))
    return TreeDecoderState(rnn_state=self.rnn_state, word_rnn_state=self.word_rnn_state, context=self.context,
                            open_nonterms=open_nonterms_copy, prev_word_state=self.prev_word_state)

class TreeDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!TreeDecoder'


  def __init__(self, yaml_context, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None, word_lstm=False):
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
    self.trg_embed_dim = trg_embed_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    plain_lstm_input = lstm_input
    # parent state + last_word_state + sibling state
    lstm_input += lstm_dim * 2
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
    #if word_lstm:
    #  self.context_projector = xnmt.linear.Linear(input_dim  = input_dim + lstm_dim * 2,
    #                                         output_dim = mlp_hidden_dim,
    #                                         model = param_col)
    #else:
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

  def initial_state(self, enc_final_states, ss_expr, decoding=False):
    """Get the initial state of the decoder given the encoder final states.

    :param enc_final_states: The encoder final states.
    :returns: An MlpSoftmaxDecoderState
    """
    rnn_state = self.fwd_lstm.initial_state()
    init_state = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(init_state)

    zeros = dy.zeros(self.init_lstm_dim)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))

    if self.set_word_lstm:
      # init_state: [c, h]
      word_rnn_state = self.word_lstm.initial_state(init_state)
      zeros = dy.zeros(self.input_dim) if self.input_feeding else None
      word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    else:
      word_rnn_state = None

    if decoding:
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, \
          open_nonterms=[OpenNonterm('ROOT', parent_state=dy.zeros(self.lstm_dim), sib_state=dy.zeros(self.lstm_dim))], \
          prev_word_state=dy.zeros(self.lstm_dim))
    else:     
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, \
              states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]))

  def add_input(self, tree_dec_state, trg_embedding, trg, trg_rule_vocab=None):
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
    if not trg_rule_vocab:
      # get parent states for this batch
      batch_size = trg_embedding.dim()[1]
      paren_tm1_states = tree_dec_state.states[trg.get_col(1)] # ((hidden_dim,), batch_size) * batch_size
      last_word_states = tree_dec_state.states[trg.get_col(2)]
      #sib_states = tree_dec_state.states[trg.get_col(4)]
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_list = []
      last_word_list = []
      #sib_state_list = []
      for i in range(batch_size):
        paren_tm1_list.append(dy.pick_batch_elem(paren_tm1_states[i], i))
        last_word_list.append(dy.pick_batch_elem(last_word_states[i], i))
        #sib_state_list.append(dy.pick_batch_elem(sib_states[i], i))
      paren_tm1_state = dy.concatenate_to_batch(paren_tm1_list)
      last_word_state = dy.concatenate_to_batch(last_word_list)
      #sib_state = dy.concatenate_to_batch(sib_state_list)

      inp = dy.concatenate([inp, paren_tm1_state, last_word_state])
      #inp = dy.concatenate([inp, paren_tm1_state, sib_state])
      # get word_rnn state
      word_rnn_state = tree_dec_state.word_rnn_state
      if self.set_word_lstm:
        word_inp = trg_embedding
        if self.input_feeding:
          word_inp = dy.concatenate([word_inp, tree_dec_state.context])
        #word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp, inv_mask=is_terminal)
        if np.count_nonzero(is_terminal) > 0:
          word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp)
        inp = dy.concatenate([inp, word_rnn_state.output()])

      rnn_state = tree_dec_state.rnn_state.add_input(inp)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, \
                           states=np.append(tree_dec_state.states, rnn_state.output()))
    else:
      cur_nonterm = tree_dec_state.open_nonterms.pop()
      rule = trg_rule_vocab[trg]
      if cur_nonterm.label != rule.lhs:
        for c in cur_nonterm:
          print c.label
      assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)
      inp = dy.concatenate([inp, cur_nonterm.parent_state, tree_dec_state.prev_word_state])
      #inp = dy.concatenate([inp, cur_nonterm.parent_state, cur_nonterm.sib_state])

      word_rnn_state = tree_dec_state.word_rnn_state
      if self.set_word_lstm:
        if len(rule.open_nonterms) == 0:
          word_inp = trg_embedding
          if self.input_feeding:
            word_inp = dy.concatenate([word_inp, tree_dec_state.context])
          word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp)
        inp = dy.concatenate([inp, word_rnn_state.output()])
 
      rnn_state = tree_dec_state.rnn_state.add_input(inp)
      # add current state to its sibling
      #if cur_nonterm.is_sibling:
      #  tree_dec_state.open_nonterms[-1].sib_state = rnn_state.output()
      # add rule to tree_dec_state.open_nonterms
      prev_word_state = tree_dec_state.prev_word_state
      open_nonterms=tree_dec_state.open_nonterms[:]
      new_open_nonterms = []
      for rhs in rule.rhs:
        if new_open_nonterms:
          new_open_nonterms[-1].is_sibling = True # sibling ends at the second to last child
        if rhs in rule.open_nonterms:
          #new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output(), sib_state=dy.zeros(self.lstm_dim)))
          new_open_nonterms.append(OpenNonterm(rhs, parent_state=rnn_state.output()))
        else:
          prev_word_state = rnn_state.output()
      new_open_nonterms.reverse()
      open_nonterms.extend(new_open_nonterms)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, \
                              open_nonterms=open_nonterms, prev_word_state=prev_word_state)

  def get_scores(self, tree_dec_state, trg_rule_vocab=None):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    #if self.set_word_lstm:
    #  h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.word_rnn_state.output(), tree_dec_state.context])))
    #else:
    h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context])))
    if not trg_rule_vocab:
      return self.vocab_projector(h_t)
    else:
      valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.open_nonterms[-1].label)
      valid_y_mask = np.ones((len(trg_rule_vocab),)) * (-1000)
      valid_y_mask[valid_y_index] = 0.
      return self.vocab_projector(h_t) + dy.inputTensor(valid_y_mask), len(valid_y_index)

  def calc_loss(self, tree_dec_state, ref_action):
    scores = self.get_scores(tree_dec_state)
    ref_word = ref_action.get_col(0)
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

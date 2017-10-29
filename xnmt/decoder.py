import dynet as dy
import numpy as np
from xnmt.serializer import Serializable
import xnmt.batcher
from xnmt.hier_model import HierarchicalModel, recursive
import xnmt.linear
from input import *
from lstm import CustomCompactLSTMBuilder

class Decoder(HierarchicalModel):
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

class MlpSoftmaxDecoderState:
  """A state holding all the information needed for MLPSoftmaxDecoder"""
  def __init__(self, rnn_state=None, context=None):
    self.rnn_state = rnn_state
    self.context = context

class MlpSoftmaxDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!MlpSoftmaxDecoder'

  def __init__(self, context, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None):
    param_col = context.dynet_param_collection.param_col
    self.param_col = param_col
    # Define dim
    lstm_dim       = lstm_dim or context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or context.default_layer_dim
    input_dim      = input_dim or context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(context, self.lstm_layers, self.lstm_dim)

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
    self.dropout = dropout or context.dropout

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

  @recursive
  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)

class OpenNonterm:
  def __init__(self, label=None, parent_state=None):
    self.label = label 
    self.parent_state = parent_state

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
    #open_nonterms_copy = []
    #for n in self.open_nonterms:
    #  open_nonterms_copy.append(OpenNonterm(n.label, n.parent_state))
    return TreeDecoderState(rnn_state=self.rnn_state, word_rnn_state=self.word_rnn_state, context=self.context, \
                            open_nonterms=self.open_nonterms[:], prev_word_state=self.prev_word_state)
  #def copy(self):
  #  if hasattr(self.tree, 'copy'):
  #    return TreeDecoderState(rnn_state=self.rnn_state, context=self.context, states=np.array(self.states), tree=self.tree.copy())
  #  else:
  #    return TreeDecoderState(rnn_state=self.rnn_state, context=self.context, states=np.array(self.states), tree=tree)

class TreeDecoder(RnnDecoder, Serializable):
  # TODO: This should probably take a softmax object, which can be normal or class-factored, etc.
  # For now the default behavior is hard coded.

  yaml_tag = u'!TreeDecoder'

  def __init__(self, context, vocab_size, layers=1, input_dim=None, lstm_dim=None,
               mlp_hidden_dim=None, trg_embed_dim=None, dropout=None,
               rnn_spec="lstm", residual_to_output=False, input_feeding=True,
               bridge=None, word_lstm=False):
    param_col = context.dynet_param_collection.param_col
    self.param_col = param_col
    # Define dim
    lstm_dim       = lstm_dim or context.default_layer_dim
    mlp_hidden_dim = mlp_hidden_dim or context.default_layer_dim
    trg_embed_dim  = trg_embed_dim or context.default_layer_dim
    input_dim      = input_dim or context.default_layer_dim
    self.input_dim = input_dim
    # Input feeding
    self.input_feeding = input_feeding
    self.lstm_dim = lstm_dim
    self.trg_embed_dim = trg_embed_dim
    lstm_input = trg_embed_dim
    if input_feeding:
      lstm_input += input_dim
    plain_lstm_input = lstm_input
    # parent state + last_word_state
    lstm_input += lstm_dim * 2
    if word_lstm:
      lstm_input += lstm_dim

    self.init_lstm_dim = lstm_input - trg_embed_dim
    # Bridge
    self.lstm_layers = layers
    self.bridge = bridge or NoBridge(context, self.lstm_layers, self.lstm_dim)

    # LSTM
    self.fwd_lstm  = RnnDecoder.rnn_from_spec(spec       = rnn_spec,
                                              num_layers = layers,
                                              input_dim  = lstm_input,
                                              hidden_dim = lstm_dim,
                                              model = param_col,
                                              residual_to_output = residual_to_output)

    self.word_lstm = word_lstm
    if word_lstm:
      self.word_lstm = RnnDecoder.rnn_from_spec(spec       = "custom",
                                                num_layers = layers,
                                                input_dim  = plain_lstm_input,
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
    self.dropout = dropout or context.dropout

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
    #zeros = dy.zeros(self.lstm_dim + self.input_dim) if self.input_feeding \
    #  else  dy.zeros(self.lstm_dim)
    zeros = dy.zeros(self.init_lstm_dim)
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]))

    if self.word_lstm:
      # init_state: [c, h]
      word_rnn_state = self.word_lstm.initial_state(init_state)
      zeros = dy.zeros(self.input_dim) if self.input_feeding else None
      word_rnn_state = word_rnn_state.add_input(dy.concatenate([ss_expr, zeros]))
    else:
      word_rnn_state = None

    if decoding:
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, \
          open_nonterms=[OpenNonterm('ROOT', parent_state=dy.zeros(self.lstm_dim))], \
          prev_word_state=dy.zeros(self.lstm_dim))
    else:
      batch_size = ss_expr.dim()[1]
      return TreeDecoderState(rnn_state=rnn_state, context=zeros, word_rnn_state=word_rnn_state, \
              states=np.array([dy.zeros((self.lstm_dim,), batch_size=batch_size)]), tree=Tree())

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
      is_terminal = trg.get_col(3, batched=False)
      paren_tm1_list = []
      last_word_list = []
      for i in range(batch_size):
        paren_tm1_list.append(dy.pick_batch_elem(paren_tm1_states[i], i))
        last_word_list.append(dy.pick_batch_elem(last_word_states[i], i))
      paren_tm1_state = dy.concatenate_to_batch(paren_tm1_list)
      last_word_state = dy.concatenate_to_batch(last_word_list)
      #terminal_mask = dy.inputTensor(np.transpose(is_terminal), batched=True)

      inp = dy.concatenate([inp, paren_tm1_state, last_word_state])
      # get word_rnn state
      if self.word_lstm:
        word_inp = trg_embedding
        if self.input_feeding:
          word_inp = dy.concatenate([word_inp, tree_dec_state.context])
        word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp, inv_mask=is_terminal)
        inp = dy.concatenate([inp, word_rnn_state.output()])
      else:
        word_rnn_state = None

      rnn_state = tree_dec_state.rnn_state.add_input(inp)
      return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, word_rnn_state=word_rnn_state, \
                           states=np.append(tree_dec_state.states, rnn_state.output()))
    else:
      # decoding time
      #lhs_node_id = tree_dec_state.tree.get_next_open_node()
      #rule = trg_rule_vocab[trg]
      #assert tree_dec_state.tree.id2n[lhs_node_id].label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, lhs_tree_node.label)
      #tree_dec_state.tree.add_rule(lhs_node_id, rule)
      #t_data = tree_dec_state.tree.get_timestep_data(lhs_node_id)

      #inp = dy.concatenate([inp] + tree_dec_state.states[t_data].tolist())
      #rnn_state = tree_dec_state.rnn_state.add_input(inp)
      #return TreeDecoderState(rnn_state=rnn_state, context=tree_dec_state.context, \
      #                      states=np.append(tree_dec_state.states, rnn_state.output()), tree=tree_dec_state.tree)
      # decoding time
      cur_nonterm = tree_dec_state.open_nonterms.pop()
      rule = trg_rule_vocab[trg]
      if cur_nonterm.label != rule.lhs:
        for c in cur_nonterm:
          print c.label
      assert cur_nonterm.label == rule.lhs, "the lhs of the current input rule %s does not match the next open nonterminal %s" % (rule.lhs, cur_nonterm.label)
      # add rule to tree_dec_state.open_nonterms
      new_open_nonterms = []
      for rhs in rule.rhs:
        if rhs in rule.open_nonterms:
          new_open_nonterms.append(OpenNonterm(rhs, parent_state=tree_dec_state.rnn_state.output()))
        else:
          tree_dec_state.prev_word_state = tree_dec_state.rnn_state.output()
      new_open_nonterms.reverse()
      tree_dec_state.open_nonterms.extend(new_open_nonterms)
      inp = dy.concatenate([inp, cur_nonterm.parent_state, tree_dec_state.prev_word_state])

      if self.word_lstm:
        if len(new_open_nonterms) == 0:
          word_inp = trg_embedding
          if self.input_feeding:
            word_inp = dy.concatenate([word_inp, tree_dec_state.context])
          tree_dec_state.word_rnn_state = tree_dec_state.word_rnn_state.add_input(word_inp)
        inp = dy.concatenate([inp, tree_dec_state.word_rnn_state.output()])

      tree_dec_state.rnn_state = tree_dec_state.rnn_state.add_input(inp)
      return tree_dec_state

  def get_scores(self, tree_dec_state, trg_rule_vocab=None):
    """Get scores given a current state.

    :param mlp_dec_state: An MlpSoftmaxDecoderState object.
    :returns: Scores over the vocabulary given this state.
    """
    h_t = dy.tanh(self.context_projector(dy.concatenate([tree_dec_state.rnn_state.output(), tree_dec_state.context])))
    if not trg_rule_vocab:
      return self.vocab_projector(h_t)
    else:
      #valid_y_index = trg_rule_vocab.rule_index_with_lhs(tree_dec_state.tree.query_open_node_label())
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

  @recursive
  def set_train(self, val):
    self.fwd_lstm.set_dropout(self.dropout if val else 0.0)

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
  def __init__(self, context, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or context.default_layer_dim
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
  def __init__(self, context, dec_layers, dec_dim = None):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim or context.default_layer_dim
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
  def __init__(self, context, dec_layers, enc_dim = None, dec_dim = None):
    param_col = context.dynet_param_collection.param_col
    self.dec_layers = dec_layers
    self.enc_dim = enc_dim or context.default_layer_dim
    self.dec_dim = dec_dim or context.default_layer_dim
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

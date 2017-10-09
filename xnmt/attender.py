import dynet as dy
from xnmt.batcher import *
from xnmt.serializer import *

class Attender(object):
  '''
  A template class for functions implementing attention.
  '''

  def __init__(self, input_dim):
    """
    :param input_dim: every attender needs an input_dim
    """
    pass

  def init_sent(self, sent):
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, state):
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')


class StandardAttender(Attender, Serializable):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  yaml_tag = u'!StandardAttender'

  def __init__(self, context, input_dim=None, state_dim=None, hidden_dim=None):
    input_dim = input_dim or context.default_layer_dim
    state_dim = state_dim or context.default_layer_dim
    hidden_dim = hidden_dim or context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = context.dynet_param_collection.param_col
    self.pW = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb = param_collection.add_parameters(hidden_dim)
    self.pU = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent = None

  def init_sent(self, sent):
    self.attention_vecs = []
    self.curr_sent = sent
    I = self.curr_sent.as_tensor()
    W = dy.parameter(self.pW)
    b = dy.parameter(self.pb)

    self.WI = dy.affine_transform([b, W, I])
    wi_dim = self.WI.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI = dy.reshape(self.WI, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V = dy.parameter(self.pV)
    U = dy.parameter(self.pU)

    h = dy.tanh(dy.colwise_add(self.WI, V * state))
    scores = dy.transpose(U * h)
    if self.curr_sent.mask is not None:
      scores = self.curr_sent.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized = dy.softmax(scores)
    self.attention_vecs.append(normalized)
    return normalized

  def calc_context(self, state):
    attention = self.calc_attention(state)
    I = self.curr_sent.as_tensor()
    return I * attention


class DoubleAttender(Attender, Serializable):
  '''
  Implements the attention model of Bahdanau et. al (2014)
  '''

  yaml_tag = u'!DoubleAttender'

  def __init__(self, context, input_dim=None, state_dim=None, hidden_dim=None):
    input_dim = input_dim or context.default_layer_dim
    state_dim = state_dim or context.default_layer_dim
    hidden_dim = hidden_dim or context.default_layer_dim
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = context.dynet_param_collection.param_col
    self.pW_mt = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV_mt = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb_mt = param_collection.add_parameters(hidden_dim)
    self.pU_mt = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent_mt = None

    self.pW_src = param_collection.add_parameters((hidden_dim, input_dim))
    self.pV_src = param_collection.add_parameters((hidden_dim, state_dim))
    self.pb_src = param_collection.add_parameters(hidden_dim)
    self.pU_src = param_collection.add_parameters((1, hidden_dim))
    self.curr_sent_src = None

  def init_sent(self, mt_sent, src_sent):
    self.attention_vecs_mt = []
    self.attention_vecs_src = []
    self.curr_sent_mt = mt_sent
    self.curr_sent_src = src_sent
    I_mt = self.curr_sent_mt.as_tensor()
    W_mt= dy.parameter(self.pW_mt)
    b_mt = dy.parameter(self.pb_mt)

    self.WI_mt = dy.affine_transform([b_mt, W_mt, I_mt])
    wi_dim = self.WI_mt.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI_mt = dy.reshape(self.WI_mt, (wi_dim[0][0], 1), batch_size=wi_dim[1])

    I_src = self.curr_sent_src.as_tensor()
    W_src = dy.parameter(self.pW_src)
    b_src = dy.parameter(self.pb_src)

    self.WI_src = dy.affine_transform([b_src, W_src, I_src])
    wi_dim = self.WI_src.dim()
    # TODO(philip30): dynet affine transform bug, should be fixed upstream
    # if the input size is "1" then the last dimension will be dropped.
    if len(wi_dim[0]) == 1:
      self.WI_src = dy.reshape(self.WI_src, (wi_dim[0][0], 1), batch_size=wi_dim[1])

  def calc_attention(self, state):
    V_mt = dy.parameter(self.pV_mt)
    U_mt = dy.parameter(self.pU_mt)

    h_mt = dy.tanh(dy.colwise_add(self.WI_mt, V_mt * state))
    scores = dy.transpose(U_mt * h_mt)
    if self.curr_sent_mt.mask is not None:
      scores = self.curr_sent_mt.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized_mt = dy.softmax(scores)
    self.attention_vecs_mt.append(normalized_mt)

    V_src = dy.parameter(self.pV_src)
    U_src = dy.parameter(self.pU_src)

    h_src = dy.tanh(dy.colwise_add(self.WI_src, V_src * state))
    scores = dy.transpose(U_src * h_src)
    if self.curr_sent_src.mask is not None:
      scores = self.curr_sent_src.mask.add_to_tensor_expr(scores, multiplicator = -100.0)
    normalized_src = dy.softmax(scores)
    self.attention_vecs_src.append(normalized_src)
    return normalized_mt, normalized_src

  def calc_context(self, state):
    attention_mt, attention_src = self.calc_attention(state)
    I_mt = self.curr_sent_mt.as_tensor()
    I_src = self.curr_sent_src.as_tensor()
    return dy.concatenate([I_mt * attention_mt, I_src * attention_src])
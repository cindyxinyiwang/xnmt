import six
from xnmt.vocab import Vocab, Rule
from input import *

class Output(object):
  '''
  A template class to represent all output.
  '''
  def __init__(self, actions=None):
    ''' Initialize an output with actions. '''
    self.actions = actions or []

  def to_string(self):
    raise NotImplementedError('All outputs must implement to_string.')

class TextOutput(Output):
  def __init__(self, actions=None, vocab=None):
    self.actions = actions or []
    self.vocab = vocab
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])

  def to_string(self, tag_set=None):
    ret = six.moves.map(lambda wi: self.vocab[wi], filter(lambda wi: wi not in self.filtered_tokens, self.actions))
    if tag_set:
      ret = [w for w in ret if w not in tag_set]
    return ret

class TreeHierOutput(Output):
  def __init__(self, actions=None, rule_vocab=None, word_vocab=None):
    self.actions = actions or []
    self.rule_vocab = rule_vocab
    self.word_vocab = word_vocab
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])
  def to_string(self):
    ret = []
    for a in self.actions:
      #if a[0] in self.filtered_tokens: continue
      if a[1]: # if is terminal
        ret.append([self.word_vocab[a[0]], a[2]])
      else:
        ret.append([self.rule_vocab[a[0]], a[2]])
    return ret

class OutputProcessor(object):
  def process_outputs(self, outputs):
    raise NotImplementedError()

class PlainTextOutputProcessor(OutputProcessor):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''
  def process_outputs(self, outputs):
    return [self.words_to_string(output.to_string()) for output in outputs]

  def words_to_string(self, word_list):
    return u" ".join(word_list)

class CcgPieceOutputProcessor(OutputProcessor):
  '''
  Handles the typical case of writing plain text,
  with one sent per line.
  '''
  def __init__(self, tag_set, merge_indicator=u"\u2581"):
    self.filtered_tokens = set([Vocab.SS, Vocab.ES])
    self.tag_set = tag_set
    self.merge_indicator = merge_indicator

  def to_string(self):
    words = six.moves.map(lambda wi: self.vocab[wi], filter(lambda wi: wi not in self.filtered_tokens, self.actions))
    words = [w for w in words if w not in self.tag_set]
    return words

  def process_outputs(self, outputs):
    return [self.words_to_string(output.to_string(self.tag_set)) for output in outputs]

  def words_to_string(self, word_list):
    return u"".join(word_list).replace(self.merge_indicator, u" ").strip()

class RuleOutputProcessor(PlainTextOutputProcessor):
  def __init__(self, piece=True, wordswitch=False):
    self.piece = piece
    self.wordswitch = wordswitch

  def words_to_string(self, rule_list):
    tree = Tree.from_rule_deriv(rule_list, wordswitch=self.wordswitch)
    return [tree.to_string(self.piece), tree.to_parse_string()]

class RuleBPEOutputProcessor(PlainTextOutputProcessor):
  def __init__(self, wordswitch=False):
    self.wordswitch = wordswitch

  def words_to_string(self, rule_list):
    tree = Tree.from_rule_deriv(rule_list, wordswitch=self.wordswitch)
    str = tree.to_string(piece=False)

    return [str.replace(u"@@ ", u""), tree.to_parse_string()]

class JoinedCharTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a single-character vocabulary and joins them to form words;
  per default, double underscores '__' are converted to spaces
  '''
  def __init__(self, space_token=u"__"):
    self.space_token = space_token

  def words_to_string(self, word_list):
    return u"".join(map(lambda s: u" " if s==self.space_token else s, word_list))

class JoinedBPETextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  '''
  def __init__(self, merge_indicator=u"@@"):
    self.merge_indicator_with_space = merge_indicator + u" "

  def words_to_string(self, word_list):
    return u" ".join(word_list).replace(self.merge_indicator_with_space, u"")

class TreeOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a Rule vocabulary and outputs the merged words;
  '''
  def __init__(self, piece=True, wordswith=True):
    self.piece = piece
    self.wordswitch = wordswith

  def words_to_string(self, word_list):
    return Tree.from_rule_deriv(word_list).to_string(self.piece)

class JoinedBPETextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  '''
  def __init__(self, merge_indicator=u"@@"):
    self.merge_indicator_with_space = merge_indicator + u" "

  def words_to_string(self, word_list):
    return u" ".join(word_list).replace(self.merge_indicator_with_space, u"")

class JoinedPieceTextOutputProcessor(PlainTextOutputProcessor):
  '''
  Assumes a bpe-based vocabulary and outputs the merged words;
  per default, the '@' postfix indicates subwords that should be merged
  '''
  def __init__(self, merge_indicator=u"\u2581"):
    self.merge_indicator = merge_indicator

  def words_to_string(self, word_list):
    return u"".join(word_list).replace(self.merge_indicator, u" ").strip()

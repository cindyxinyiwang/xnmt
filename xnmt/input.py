import numpy as np
import os
import io
import six
import re
from collections import defaultdict
from six.moves import zip_longest, map
from xnmt.serializer import Serializable
from xnmt.vocab import *
###### Classes representing single inputs

class Input(object):
  """
  A template class to represent all inputs.
  """
  def __len__(self):
    raise NotImplementedError("__len__() must be implemented by Input subclasses")

  def __getitem__(self):
    raise NotImplementedError("__getitem__() must be implemented by Input subclasses")

  def get_padded_sent(self, token, pad_len):
    raise NotImplementedError("get_padded_sent() must be implemented by Input subclasses")

class SimpleSentenceInput(Input):
  """
  A simple sent, represented as a list of tokens
  """
  def __init__(self, words):
    self.words = words

  def __len__(self):
    return len(self.words)

  def __getitem__(self, key):
    return self.words[key]

  def get_padded_sent(self, token, pad_len):
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([token] * pad_len)
    return SimpleSentenceInput(new_words)

  def __str__(self):
    return " ".join(six.moves.map(str, self.words))

class SentenceInput(SimpleSentenceInput):
  def __init__(self, words):
    super(SentenceInput, self).__init__(words)
    self.annotation = []

  def annotate(self, key, value):
    self.__dict__[key] = value

class ArrayInput(Input):
  """
  A sent based on a single numpy array; first dimension contains tokens
  """
  def __init__(self, nparr):
    self.nparr = nparr

  def __len__(self):
    return self.nparr.shape[1] if len(self.nparr.shape) >= 2 else 1

  def __getitem__(self, key):
    return self.nparr.__getitem__(key)

  def get_padded_sent(self, token, pad_len):
    if pad_len == 0:
      return self
    new_nparr = np.append(self.nparr, np.zeros((self.nparr.shape[0], pad_len)), axis=1)
    return ArrayInput(new_nparr)

  def get_array(self):
    return self.nparr

###### Classes that will read in a file and turn it into an input

class InputReader(object):
  def read_sents(self, filename, filter_ids=None):
    """
    :param filename: data file
    :param filter_ids: only read sentences with these ids (0-indexed)
    :returns: iterator over sentences from filename
    """
    raise RuntimeError("Input readers must implement the read_sents function")

  def count_sents(self, filename):
    """
    :param filename: data file
    :returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def freeze(self):
    pass

class BaseTextReader(InputReader):
  def count_sents(self, filename):
    i = 0
    with io.open(filename, encoding='utf-8') as f:
      for _ in f:
        i+=1
    return i

  def iterate_filtered(self, filename, filter_ids=None):
    """
    :param filename: data file (text file)
    :param filter_ids:
    :returns: iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    sent_count = 0
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)
    with io.open(filename, encoding='utf-8') as f:
      for line in f:
        if filter_ids is None or sent_count in filter_ids:
          yield line
        sent_count += 1
        if max_id is not None and sent_count > max_id:
          break

class PlainTextReader(BaseTextReader, Serializable):
  """
  Handles the typical case of reading plain text files,
  with one sent per line.
  """
  yaml_tag = u'!PlainTextReader'
  def __init__(self, vocab=None):
    self.vocab = vocab
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = Vocab()
    return map(lambda l: SimpleSentenceInput([self.vocab.convert(word) for word in l.strip().split()] + \
                                                      [self.vocab.convert(Vocab.ES_STR)]),
               self.iterate_filtered(filename, filter_ids))

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param("vocab", self.vocab)

  def vocab_size(self):
    return len(self.vocab)

class ContVecReader(InputReader, Serializable):
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".npz" file, which consists of multiply ".npy" files, each
  corresponding to a single sequence of continuous features. This can be
  created in two ways:
  * Use the builtin function numpy.savez_compressed()
  * Create a bunch of .npy files, and run "zip" on them to zip them into an archive.

  The file names should be named XXX_0, XXX_1, etc., where the final number after the underbar
  indicates the order of the sequence in the corpus. This is done automatically by
  numpy.savez_compressed(), in which case the names will be arr_0, arr_1, etc.

  Each numpy file will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,word_ind] if transpose=False
  * sents[sent_id][word_ind,feat_ind] if transpose=True
  """
  yaml_tag = u"!ContVecReader"

  def __init__(self, transpose=False):
    self.transpose = transpose

  def read_sents(self, filename, filter_ids=None):
    npzFile = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npzKeys = sorted(npzFile.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      npzKeys = [npzKeys[i] for i in filter_ids]
    for idx, key in enumerate(npzKeys):
      inp = npzFile[key]
      if self.transpose:
        inp = inp.transpose()
      if idx % 1000 == 999:
        print("Read {} lines ({:.2f}%) of {} at {}".format(idx+1, float(idx+1)/len(npzKeys)*100, filename, key))
      yield ArrayInput(inp)
    npzFile.close()

  def count_sents(self, filename):
    npzFile = np.load(filename, mmap_mode="r") # for counting sentences, only read the index
    l = len(npzFile.files)
    npzFile.close()
    return l

class IDReader(BaseTextReader, Serializable):
  """
  Handles the case where we need to read in a single ID (like retrieval problems)
  """
  yaml_tag = u"!IDReader"

  def read_sents(self, filename, filter_ids=None):
    return map(lambda l: int(l.strip()), self.iterate_filtered(filename, filter_ids))

###### CorpusParser

class CorpusParser:
  """A class that can read in corpora for training and testing"""

  def read_training_corpus(self, training_corpus):
    """Read in the training corpus"""
    raise RuntimeError("CorpusParsers must implement read_training_corpus to read in the training/dev corpora")


class BilingualCorpusParser(CorpusParser, Serializable):
  """A class that reads in bilingual corpora, consists of two InputReaders"""

  yaml_tag = u"!BilingualCorpusParser"
  def __init__(self, src_reader, trg_reader, max_src_len=None, max_trg_len=None,
               max_num_train_sents=None, max_num_dev_sents=None, sample_train_sents=None):
    """
    :param src_reader: InputReader for source side
    :param trg_reader: InputReader for target side
    :param max_src_len: filter pairs longer than this on the source side
    :param max_src_len: filter pairs longer than this on the target side
    :param max_num_train_sents: only read the first n training sentences
    :param max_num_dev_sents: only read the first n dev sentences
    :param sample_train_sents: sample n sentences without replacement from the training corpus (should probably be used with a prespecified vocab)
    """
    self.src_reader = src_reader
    self.trg_reader = trg_reader
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len
    self.max_num_train_sents = max_num_train_sents
    self.max_num_dev_sents = max_num_dev_sents
    self.sample_train_sents = sample_train_sents
    self.train_src_len, self.train_trg_len = None, None
    self.dev_src_len, self.dev_trg_len = None, None
    if max_num_train_sents is not None and sample_train_sents is not None: raise RuntimeError("max_num_train_sents and sample_train_sents are mutually exclusive!")

  def read_training_corpus(self, training_corpus):
    training_corpus.train_src_data = []
    training_corpus.train_trg_data = []
    if self.sample_train_sents:
      self.train_src_len = self.src_reader.count_sents(training_corpus.train_src)
      self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
      if self.train_src_len != self.train_trg_len: raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len, self.train_trg_len))
      self.sample_train_sents = int(self.sample_train_sents)
      filter_ids = np.random.choice(self.train_src_len, self.sample_train_sents, replace=False)
    elif self.max_num_train_sents:
      self.train_src_len = self.src_reader.count_sents(training_corpus.train_src)
      self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
      if self.train_src_len != self.train_trg_len: raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len, self.train_trg_len))
      filter_ids = list(range(min(self.max_num_train_sents, self.train_trg_len)))
    else:
      filter_ids = None
    src_train_iterator = self.src_reader.read_sents(training_corpus.train_src, filter_ids)
    trg_train_iterator = self.trg_reader.read_sents(training_corpus.train_trg, filter_ids)
    for src_sent, trg_sent in six.moves.zip_longest(src_train_iterator, trg_train_iterator):
      if src_sent is None or trg_sent is None:
        raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (self.train_src_len or self.src_reader.count_sents(training_corpus.train_src), self.train_trg_len or self.trg_reader.count_sents(training_corpus.train_trg)))
      src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
      trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
      if src_len_ok and trg_len_ok:
        training_corpus.train_src_data.append(src_sent)
        training_corpus.train_trg_data.append(trg_sent)

    self.src_reader.freeze()
    self.trg_reader.freeze()

    training_corpus.dev_src_data = []
    training_corpus.dev_trg_data = []
    if self.max_num_dev_sents:
      self.dev_src_len = self.dev_src_len or self.src_reader.count_sents(training_corpus.dev_src)
      self.dev_trg_len = self.dev_trg_len or self.trg_reader.count_sents(training_corpus.dev_trg)
      if self.dev_src_len != self.dev_trg_len: raise RuntimeError("dev src sentences don't match trg sentences: %s != %s!" % (self.dev_src_len, self.dev_trg_len))
      filter_ids = list(range(min(self.max_num_dev_sents, self.dev_src_len)))
    else:
      filter_ids = None

    src_dev_iterator = self.src_reader.read_sents(training_corpus.dev_src, filter_ids)
    trg_dev_iterator = self.trg_reader.read_sents(training_corpus.dev_trg, filter_ids)
    for src_sent, trg_sent in six.moves.zip_longest(src_dev_iterator, trg_dev_iterator):
      if src_sent is None or trg_sent is None:
        raise RuntimeError("dev src sentences don't match target trg: %s != %s!" % (self.src_reader.count_sents(training_corpus.dev_src), self.dev_trg_len), self.trg_reader.count_sents(training_corpus.dev_trg))
      src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
      trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
      if src_len_ok and trg_len_ok:
        training_corpus.dev_src_data.append(src_sent)
        training_corpus.dev_trg_data.append(trg_sent)

class TreeInput(Input):
  """
  A tree input, represented as a list of rule indices and other information
  each item in words is a list of data
  """
  def __init__(self, words):
    self.words = words

  def __len__(self):
    return len(self.words)

  def __getitem__(self, key):
    return self.words[key]

  def get_padded_sent(self, token, pad_len):
    if pad_len == 0:
      return self
    new_words = list(self.words)
    new_words.extend([[token]*2 ] * pad_len)
    return TreeInput(new_words)

  def __str__(self):
    return " ".join(six.moves.map(str, self.words))

class Tree(object):
  """A class that represents a parse tree"""

  yaml_tag = u"!Tree"
  def __init__(self, string, children):
    self.label = string 
    self.children = children
    for c in children:
      if hasattr(c, "set_parent"):
        c.set_parent(self)
    self._parent = None
    self.timestep = -1

  @classmethod
  def from_rule_deriv(cls, derivs):
    r0 = derivs[0]
    root = Tree(r0.lhs, [child for child in r0.rhs if not child in r0.open_nonterms ])
    if len(r0.open_nonterms) == 0:
      #assert len(derivs) == 1
      return root 
    stack_tree = [ root ]
    stack_children_left = [len(r0.open_nonterms)]
    for r in derivs[1:]:
      p_tree, p_children_left = stack_tree.pop(), stack_children_left.pop()
      new_tree = Tree(r.lhs, [child for child in r.rhs if not child in r.open_nonterms])
      p_tree.children.append(new_tree)
      new_tree._parent = p_tree
      p_children_left -= 1
      if p_children_left > 0:
        stack_tree.append(p_tree)
        stack_children_left.append(p_children_left)
      if len(r.open_nonterms) > 0:
        stack_tree.append(new_tree)
        stack_children_left.append(len(r.open_nonterms))
    return root

  def __str__(self):
    c_str = []
    stack = [self]
    add_paren_len = [0]
    while stack:
      while add_paren_len[-1] == len(stack):
        assert len(c_str) > 0
        add_paren_len.pop()
        c_str[-1] += ")"
      cur= stack.pop()
      c_str.append("(" + cur.label)
      for c in reversed(cur.children):
        if type(c) == str or type(c) == unicode:
          c_str.append(c)
        else:
          stack.append(c)
          add_paren_len.append(len(stack)-1)
    c_str[-1] += ")" * len(add_paren_len)
    return " ".join(c_str)

  def to_sentence(self, piece=False):
    '''
    convert subtree into the sentence it represents
    '''
    toks = []
    
    stack = [self]
    add_space_len = []
    while stack:
      #sub_word = True
      init_stack_len = len(stack)
      if add_space_len and add_space_len[-1] == len(stack):
        toks.append(" ")
        add_space_len.pop()
      cur = stack.pop()
      for c in reversed(cur.children):
        if type(c) == str or type(c) == unicode:
          toks.append(c)
        else:
          stack.append(c)
      if not "_sub" in cur.label and not piece:
        add_space_len.append(init_stack_len-1)
    if not piece:
      return "".join(toks)
    else:
      return " ".join(toks)

  def parent(self):
    return self._parent

  def set_parent(self, new_parent):
    self._parent = new_parent

  def set_timestep(self, t, t2n=None):
    '''
    initialize timestep for each node
    '''
    self.timestep = t 
    if not t2n is None:
      assert self.timestep == len(t2n)
      t2n.append(self)
    for c in self.children:
      if hasattr(c, 'set_timestep'):
        t = c.set_timestep(t+1, t2n)
    return t

  def _get_data_helper(self, deriv, paren_t):
    ''' deprecated method '''
    children = []
    open_nonterms = []
    for c in self.children:
      if type(c) == str or type(c) == unicode:
        children.append(c)
      else:
        children.append(c.label)
        open_nonterms.append(c.label)
    #print self
    #deriv.append(self.label + '|||' + " ".join(children))
    deriv.append(Rule(self.label, children, open_nonterms))
    if self.parent():
      paren_t.append(self.parent().timestep)
    else:
      paren_t.append(0)
    for c in self.children:
      if hasattr(c, 'get_data'):
        c.get_data(deriv, paren_t)

  def get_data_root(self, rule_vocab):
    self.t2n = []
    self.set_timestep(0, self.t2n)
    data = []
    for t in xrange(len(self.t2n)):
      node = self.t2n[t]
      children, open_nonterms = [], []
      for c in node.children:
        if type(c) == str or type(c) == unicode:
          children.append(c)
        else:
          children.append(c.label)
      paren_t = 0 if not node.parent() else node.parent().timestep
      data.append([rule_vocab.convert(Rule(node.label, children, open_nonterms)), paren_t])
    return data

class TreeReader(BaseTextReader, Serializable):
  """
  Reads in parse tree file, with one line per 
  parse tree. The vocab object has to be a RuleVocab
  """
  yaml_tag = u'!TreeReader'
  def __init__(self, vocab=None):
    self.vocab = vocab
    if vocab is not None:
      self.vocab.freeze()
      self.vocab.set_unk(Vocab.UNK_STR)

  def read_sents(self, filename, filter_ids=None):
    if self.vocab is None:
      self.vocab = RuleVocab()
    for line in self.iterate_filtered(filename, filter_ids):
      tree = parse_root(tokenize(line))
      yield TreeInput(tree.get_data_root(self.vocab) + [ [self.vocab.convert(Vocab.ES_STR)]*2 ]) 

  def freeze(self):
    self.vocab.freeze()
    self.vocab.set_unk(Vocab.UNK_STR)
    self.overwrite_serialize_param("vocab", self.vocab)

  def vocab_size(self):
    return len(self.vocab)

# Tokenize a string.
# Tokens yielded are of the form (type, string)
# Possible values for 'type' are '(', ')' and 'WORD'
def tokenize(s):
  toks = re.compile(' +|[^() ]+|[()]')
  for match in toks.finditer(s):
    s = match.group(0)
    if s[0] == ' ':
      continue
    if s[0] in '()':
      yield (s, s)
    else:
      yield ('WORD', s)


# Parse once we're inside an opening bracket.
def parse_inner(toks):
  ty, name = next(toks)
  if ty != 'WORD': raise ParseError
  children = []
  while True:
    ty, s = next(toks)
    #print ty, s
    if ty == '(':
      children.append(parse_inner(toks))
    elif ty == ')':
      return Tree(name, children)
    else:
      children.append(s)

# Parse this grammar:
# ROOT ::= '(' INNER
# INNER ::= WORD ROOT* ')'
# WORD ::= [A-Za-z]+
def parse_root(toks):

  ty, _ = next(toks)
  if ty != '(': raise ParseError
  return parse_inner(toks)
###### Obsolete Functions

# TODO: The following doesn't follow the current API. If it is necessary, it should be retooled
# class MultilingualAlignedCorpusReader(object):
#     """Handles the case of reading TED talk files
#     """
#
#     def __init__(self, corpus_path, vocab=None, delimiter='\t', trg_token=True, bilingual=True,
#                  lang_dict={'src': ['fr'], 'trg': ['en']}, zero_shot=False, eval_lang_dict=None):
#
#         self.empty_line_flag = '__NULL__'
#         self.corpus_path = corpus_path
#         self.delimiter = delimiter
#         self.bilingual = bilingual
#         self.lang_dict = lang_dict
#         self.lang_set = set()
#         self.trg_token = trg_token
#         self.zero_shot = zero_shot
#         self.eval_lang_dict = eval_lang_dict
#
#         for list_ in self.lang_dict.values():
#             for lang in list_:
#                 self.lang_set.add(lang)
#
#         self.data = dict()
#         self.data['train'] = self.read_aligned_corpus(split_type='train')
#         self.data['test'] = self.read_aligned_corpus(split_type='test')
#         self.data['dev'] = self.read_aligned_corpus(split_type='dev')
#
#
#     def read_data(self, file_loc_):
#         data_list = list()
#         with open(file_loc_) as fp:
#             for line in fp:
#                 try:
#                     text = line.strip()
#                 except IndexError:
#                     text = self.empty_line_flag
#                 data_list.append(text)
#         return data_list
#
#
#     def filter_text(self, dict_):
#         if self.trg_token:
#             field_index = 1
#         else:
#             field_index = 0
#         data_dict = defaultdict(list)
#         list1 = dict_['src']
#         list2 = dict_['trg']
#         for sent1, sent2 in zip(list1, list2):
#             try:
#                 src_sent = ' '.join(sent1.split()[field_index: ])
#             except IndexError:
#                 src_sent = '__NULL__'
#
#             if src_sent.find(self.empty_line_flag) != -1:
#                 continue
#
#             elif sent2.find(self.empty_line_flag) != -1:
#                 continue
#
#             else:
#                 data_dict['src'].append(sent1)
#                 data_dict['trg'].append(sent2)
#         return data_dict
#
#
#     def read_sents(self, split_type, data_type):
#         return self.data[split_type][data_type]
#
#
#     def save_file(self, path_, split_type, data_type):
#         with open(path_, 'w') as fp:
#             for line in self.data[split_type][data_type]:
#                 fp.write(line + '\n')
#
#
#     def add_trg_token(self, list_, lang_id):
#         new_list = list()
#         token = '__' + lang_id + '__'
#         for sent in list_:
#             new_list.append(token + ' ' + sent)
#         return new_list
#
#     def read_aligned_corpus(self, split_type='train'):
#
#         split_type_path = os.path.join(self.corpus_path, split_type)
#         data_dict = defaultdict(list)
#
#         if self.zero_shot:
#             if split_type == "train":
#                 iterable = zip(self.lang_dict['src'], self.lang_dict['trg'])
#             else:
#                 iterable = zip(self.eval_lang_dict['src'], self.eval_lang_dict['trg'])
#
#         elif self.bilingual:
#             iterable = itertools.product(self.lang_dict['src'], self.lang_dict['trg'])
#
#         for s_lang, t_lang in iterable:
#                 for talk_dir in os.listdir(split_type_path):
#                     dir_path = os.path.join(split_type_path, talk_dir)
#
#                     talk_lang_set = set([l.split('.')[0] for l in os.listdir(dir_path)])
#
#                     if s_lang not in talk_lang_set or t_lang not in talk_lang_set:
#                         continue
#
#                     for infile in os.listdir(dir_path):
#                         lang = os.path.splitext(infile)[0]
#
#                         if lang in self.lang_set:
#                             file_path = os.path.join(dir_path, infile)
#                             text = self.read_data(file_path)
#
#                             if lang == s_lang:
#                                 if self.trg_token:
#                                     text = self.add_trg_token(text, t_lang)
#                                     data_dict['src'] += text
#                                 else:
#                                     data_dict['src'] += text
#
#                             elif lang == t_lang:
#                                 data_dict['trg'] += text
#
#         new_data_dict = self.filter_text(data_dict)
#         return new_data_dict
#
#
# if __name__ == "__main__":
#
#     # Testing the code
#     data_path = "/home/devendra/Desktop/Neural_MT/scrapped_ted_talks_dataset/web_data_temp"
#     zs_train_lang_dict={'src': ['pt-br', 'en'], 'trg': ['en', 'es']}
#     zs_eval_lang_dict = {'src': ['pt-br'], 'trg': ['es']}
#
#     obj = MultilingualAlignedCorpusReader(corpus_path=data_path, lang_dict=zs_train_lang_dict, trg_token=True,
#                                           eval_lang_dict=zs_eval_lang_dict, zero_shot=True, bilingual=False)
#
#
#     #src_test_list = obj.read_sents(split_type='test', data_type='src')
#     #trg_test_list = obj.read_sents(split_type='test', data_type='trg')
#
#     #print len(src_test_list)
#     #print len(trg_test_list)
#
#     #for sent_s, sent_t in zip(src_test_list, trg_test_list):
#     #    print sent_s, "\t", sent_t
#
#     obj.save_file("../ted_sample/zs_s.train", split_type='train', data_type='src')
#     obj.save_file("../ted_sample/zs_t.train", split_type='train', data_type='trg')
#
#     obj.save_file("../ted_sample/zs_s.test", split_type='test', data_type='src')
#     obj.save_file("../ted_sample/zs_t.test", split_type='test', data_type='trg')
#
#     obj.save_file("../ted_sample/zs_s.dev", split_type='dev', data_type='src')
#     obj.save_file("../ted_sample/zs_t.dev", split_type='dev', data_type='trg')

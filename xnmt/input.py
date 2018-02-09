import numpy as np
import os
import io
import six
import re
from collections import deque
from six.moves import zip_longest, map
from xnmt.serializer import Serializable
from xnmt.vocab import *
import codecs


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
        filename = filename.split(',')[0]
        with io.open(filename, encoding='utf-8') as f:
            for _ in f:
                i += 1
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
        print filename
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
        self.overwrite_serialize_param(u"vocab", self.vocab)

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
                print(
                "Read {} lines ({:.2f}%) of {} at {}".format(idx + 1, float(idx + 1) / len(npzKeys) * 100, filename,
                                                             key))
            yield ArrayInput(inp)
        npzFile.close()

    def count_sents(self, filename):
        npzFile = np.load(filename, mmap_mode="r")  # for counting sentences, only read the index
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
        if max_num_train_sents is not None and sample_train_sents is not None: raise RuntimeError(
            "max_num_train_sents and sample_train_sents are mutually exclusive!")

    def read_training_corpus(self, training_corpus):
        training_corpus.train_src_data = []
        training_corpus.train_trg_data = []

        if self.sample_train_sents:
            self.train_src_len = self.src_reader.count_sents(training_corpus.train_src)
            self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
            if self.train_src_len != self.train_trg_len: raise RuntimeError(
                "training src sentences don't match trg sentences: %s != %s!" % (
                self.train_src_len, self.train_trg_len))
            self.sample_train_sents = int(self.sample_train_sents)
            filter_ids = np.random.choice(self.train_src_len, self.sample_train_sents, replace=False)
        elif self.max_num_train_sents:
            self.train_src_len = self.src_reader.count_sents(training_corpus.train_src)
            self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
            if self.train_src_len != self.train_trg_len: raise RuntimeError(
                "training src sentences don't match trg sentences: %s != %s!" % (
                self.train_src_len, self.train_trg_len))
            filter_ids = list(range(min(self.max_num_train_sents, self.train_trg_len)))
        else:
            self.train_trg_len = self.trg_reader.count_sents(training_corpus.train_trg)
            filter_ids = None
        src_train_iterator = self.src_reader.read_sents(training_corpus.train_src, filter_ids)
        trg_train_iterator = self.trg_reader.read_sents(training_corpus.train_trg, filter_ids)
        if training_corpus.train_len_file:
            train_trg_data_len = []
            training_corpus.train_trg_data_len = []
            sent_count = 0
            max_id = None
            if filter_ids is not None:
                max_id = max(filter_ids)
                filter_ids = set(filter_ids)
            with io.open(training_corpus.train_len_file, encoding='utf-8') as f:
                for line in f:
                    if filter_ids is None or sent_count in filter_ids:
                        #yield line
                        train_trg_data_len.append(len(line.split()))
                    sent_count += 1
                    if max_id is not None and sent_count > max_id:
                        break
        else:
            train_trg_data_len = [0 for i in range(self.train_trg_len)]
        for src_sent, trg_sent, trg_len in six.moves.zip_longest(src_train_iterator, trg_train_iterator, train_trg_data_len):
            if src_sent is None or trg_sent is None:
                raise RuntimeError("training src sentences don't match trg sentences: %s != %s!" % (
                self.train_src_len or self.src_reader.count_sents(training_corpus.train_src),
                self.train_trg_len or self.trg_reader.count_sents(training_corpus.train_trg)))
            src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
            trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
            if src_len_ok and trg_len_ok:
                training_corpus.train_src_data.append(src_sent)
                training_corpus.train_trg_data.append(trg_sent)
                if training_corpus.train_len_file:
                    training_corpus.train_trg_data_len.append(trg_len)

        self.src_reader.freeze()
        self.trg_reader.freeze()

        training_corpus.dev_src_data = []
        training_corpus.dev_trg_data = []
        if self.max_num_dev_sents:
            self.dev_src_len = self.dev_src_len or self.src_reader.count_sents(training_corpus.dev_src)
            self.dev_trg_len = self.dev_trg_len or self.trg_reader.count_sents(training_corpus.dev_trg)
            if self.dev_src_len != self.dev_trg_len: raise RuntimeError(
                "dev src sentences don't match trg sentences: %s != %s!" % (self.dev_src_len, self.dev_trg_len))
            filter_ids = list(range(min(self.max_num_dev_sents, self.dev_src_len)))
        else:
            self.dev_trg_len = self.dev_trg_len or self.trg_reader.count_sents(training_corpus.dev_trg)
            filter_ids = None

        src_dev_iterator = self.src_reader.read_sents(training_corpus.dev_src, filter_ids)
        trg_dev_iterator = self.trg_reader.read_sents(training_corpus.dev_trg, filter_ids)
        if training_corpus.dev_len_file:
            dev_trg_data_len = []
            training_corpus.dev_trg_data_len = []
            sent_count = 0
            max_id = None
            if filter_ids is not None:
                max_id = max(filter_ids)
                filter_ids = set(filter_ids)
            with io.open(training_corpus.dev_len_file, encoding='utf-8') as f:
                for line in f:
                    if filter_ids is None or sent_count in filter_ids:
                        #yield line
                        dev_trg_data_len.append(len(line.split()))
                    sent_count += 1
                    if max_id is not None and sent_count > max_id:
                        break
        else:
            dev_trg_data_len = [0 for i in range(self.dev_trg_len)]
        for src_sent, trg_sent, trg_len in six.moves.zip_longest(src_dev_iterator, trg_dev_iterator, dev_trg_data_len):
            if src_sent is None or trg_sent is None:
                raise RuntimeError("dev src sentences don't match target trg: %s != %s!" % (
                self.src_reader.count_sents(training_corpus.dev_src), self.dev_trg_len),
                                   self.trg_reader.count_sents(training_corpus.dev_trg))
            src_len_ok = self.max_src_len is None or len(src_sent) <= self.max_src_len
            trg_len_ok = self.max_trg_len is None or len(trg_sent) <= self.max_trg_len
            if src_len_ok and trg_len_ok:
                training_corpus.dev_src_data.append(src_sent)
                training_corpus.dev_trg_data.append(trg_sent)
                if training_corpus.dev_len_file:
                    training_corpus.dev_trg_data_len.append(trg_len)


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
        new_words.extend([[token] * 6] * pad_len)
        return TreeInput(new_words)

    def __str__(self):
        return " ".join(six.moves.map(str, self.words))


class TreeReader(BaseTextReader, Serializable):
    """
  Reads in parse tree file, with one line per 
  parse tree. The vocab object has to be a RuleVocab
  """
    yaml_tag = u'!TreeReader'

    def __init__(self, vocab=None, word_vocab=None, binarize=True, del_preterm_POS=False,
                 read_word=False, merge=False, merge_level=-1, add_eos=True):
        self.vocab = vocab
        self.binarize = binarize
        self.del_preterm_POS = del_preterm_POS
        self.word_vocab = word_vocab
        self.read_word = read_word
        self.merge = merge
        self.merge_level = merge_level
        self.add_eos = add_eos
        if vocab is not None:
            self.vocab.freeze()
            self.vocab.set_unk(Vocab.UNK_STR)
        if word_vocab is not None:
            self.word_vocab.freeze()
            self.word_vocab.set_unk(Vocab.UNK_STR)

    def read_sents(self, filename, filter_ids=None):
        if self.vocab is None:
            self.vocab = RuleVocab()
        if self.read_word and self.word_vocab is None:
            self.word_vocab = Vocab()
        filename = filename.split(',')
        if len(filename) > 1:
            for line, sent_piece in self.iterate_filtered_double(filename[0], filename[1], filter_ids):
                tree = Tree(parse_root(tokenize(line)))
                if self.del_preterm_POS:
                    remove_preterminal_POS(tree.root)
                split_sent_piece(tree.root, sent_piece_segs(sent_piece), 0)
                if self.merge:
                    merge_tags(tree.root)
                if self.merge_level > 0:
                    merge_depth(tree.root, self.merge_level, 0)
                # add x after bpe
                if self.word_vocab:
                    add_preterminal_wordswitch(tree.root, self.add_eos)
                    if self.binarize:
                        tree.root = right_binarize(tree.root, self.read_word)
                else:
                    if self.binarize:
                        tree.root = right_binarize(tree.root, self.read_word)
                    add_preterminal(tree.root)
                tree.reset_timestep()
                yield TreeInput(tree.get_data_root(self.vocab, self.word_vocab))
        else:
            for line in self.iterate_filtered(filename[0], filter_ids):
                tree = Tree(parse_root(tokenize(line)))
                if self.del_preterm_POS:
                    remove_preterminal_POS(tree.root)
                if self.merge:
                    merge_tags(tree.root)
                if self.merge_level:
                    merge_depth(tree.root, self.merge_level, 0)
                if self.word_vocab:
                    add_preterminal_wordswitch(tree.root, self.add_eos)
                    if self.binarize:
                        tree.root = right_binarize(tree.root, self.read_word)
                else:
                    if self.binarize:
                        tree.root = right_binarize(tree.root, self.read_word)
                    add_preterminal(tree.root)
                tree.reset_timestep()
                yield TreeInput(tree.get_data_root(self.vocab, self.word_vocab))

    def freeze(self):
        if self.word_vocab:
            self.word_vocab.freeze()
            self.word_vocab.set_unk(Vocab.UNK_STR)
            self.overwrite_serialize_param('word_vocab', self.word_vocab)
        self.vocab.freeze()
        self.vocab.set_unk(Vocab.UNK_STR)
        self.overwrite_serialize_param("vocab", self.vocab)
        self.vocab.tag_vocab.freeze()
        self.vocab.tag_vocab.set_unk(Vocab.UNK_STR)
        self.vocab.overwrite_serialize_param("tag_vocab", self.vocab.tag_vocab)

    def vocab_size(self):
        return len(self.vocab)

    def tag_vocab_size(self):
        return len(self.vocab.tag_vocab)

    def word_vocab_size(self):
        return len(self.word_vocab)
    def iterate_filtered_double(self, file1, file2, filter_ids=None):
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
        with io.open(file1, encoding='utf-8') as f1:
            with io.open(file2, encoding='utf-8') as f2:
                for line1, line2 in zip(f1, f2):
                    if filter_ids is None or sent_count in filter_ids:
                        yield line1, line2
                    sent_count += 1
                    if max_id is not None and sent_count > max_id:
                        break


class TreeNode(object):
    """A class that represents a tree node object """

    def __init__(self, string, children, timestep=-1, id=-1, last_word_t=0):
        self.label = string
        self.children = children
        for c in children:
            if hasattr(c, "set_parent"):
                c.set_parent(self)
        self._parent = None
        self.timestep = timestep
        self.id = id
        self.last_word_t = last_word_t
        self.frontir_label = None

    def is_preterminal(self):
        #return len(self.children) == 1 and (not hasattr(self.children[0], 'is_preterminal'))
        for c in self.children:
            if hasattr(c, 'is_preterminal'): return False
        return True

    def to_parse_string(self):
        c_str = []
        stack = [self]
        while stack:
            cur = stack.pop()
            while not hasattr(cur, 'label'):
                c_str.append(cur)
                if not stack: break
                cur = stack.pop()
            if not hasattr(cur, 'children'): break
            stack.append(u')')
            for c in reversed(cur.children):
                stack.append(c)
            stack.append(u'({} '.format(cur.label) )
        return u"".join(c_str)

    def to_string(self, piece=True):
        '''
    convert subtree into the sentence it represents
    '''
        toks = []
        stack = [self]
        while stack:
            cur = stack.pop()
            while not hasattr(cur, 'label'):
                toks.append(cur)
                if not stack: break
                cur = stack.pop()
            if not hasattr(cur, 'children'): break
            for c in reversed(cur.children):
                stack.append(c)
        if not piece:
            return u" ".join(toks)
        else:
            return u"".join(toks).replace(u'\u2581', u' ').strip()

    def parent(self):
        return self._parent

    def set_parent(self, new_parent):
        self._parent = new_parent

    def add_child(self, child, id2n=None, last_word_t=0):
        self.children.append(child)
        if hasattr(child, "set_parent"):
            child._parent = self
            child.last_word_t = last_word_t
            if id2n:
                child.id = len(id2n)
                id2n[child.id] = child
                return child.id
        return -1

    def copy(self):
        new_node = TreeNode(self.label, [])
        for c in self.children:
            if hasattr(c, 'copy'):
                new_node.add_child(c.copy())
            else:
                new_node.add_child(c)
        return new_node

    def frontir_nodes(self):
        frontir = []
        for c in self.children:
            if hasattr(c, 'children'):
                if len(c.children) == 0:
                    frontir.append(c)
                else:
                    frontir.extend(c.frontir_nodes())

        return frontir

    def leaf_nodes(self):
        leaves = []
        for c in self.children:
            if hasattr(c, 'children'):
                leaves.extend(c.leaf_nodes())
            else:
                leaves.append(c)
        return leaves

    def get_leaf_lens(self, len_dict):
        if self.is_preterminal():
            l = self.leaf_nodes()
            #if len(l) > 10:
            #    print l, len(l)
            len_dict[len(l)] += 1
            return
        for c in self.children:
            if hasattr(c, 'is_preterminal'):
                c.get_leaf_lens(len_dict)

    def set_timestep(self, t, t2n=None, id2n=None, last_word_t=0, sib_t=0, open_stack=[]):
        '''
    initialize timestep for each node
    '''
        self.timestep = t
        self.last_word_t = last_word_t
        self.sib_t = sib_t
        next_word_t = last_word_t
        if not t2n is None:
            assert self.timestep == len(t2n)
            assert t not in t2n
            t2n[t] = self
        if not id2n is None:
            self.id = t
            id2n[t] = self
        sib_t = 0

        assert self.label == open_stack[-1]
        open_stack.pop()
        new_open_label = []
        for c in self.children:
            if hasattr(c, 'set_timestep'):
                new_open_label.append(c.label)
        new_open_label.reverse()
        open_stack.extend(new_open_label)

        if open_stack:
            self.frontir_label = open_stack[-1]
        else:
            self.frontir_label = Vocab.ES_STR
        c_t = t
        for c in self.children:
            #c_t = t + 1  # time of current child
            if hasattr(c, 'set_timestep'):
                c_t = t + 1
                t, next_word_t = c.set_timestep(c_t, t2n, id2n, next_word_t, sib_t, open_stack)
            else:
                next_word_t = t
            sib_t = c_t
        return t, next_word_t


class Tree(object):
    """A class that represents a parse tree"""

    yaml_tag = u"!Tree"

    def __init__(self, root=None, sent_piece=None, binarize=False):
        self.id2n = {}
        self.t2n = {}
        self.open_nonterm_ids = []
        self.last_word_t = -1
        if root:
            self.root = TreeNode(u'XXX', [root])
            #if sent_piece:
            #    split_sent_piece(self.root, sent_piece_segs(sent_piece), 0)
            #if binarize:
            #    self.root = right_binarize(self.root)
            #self.root.set_timestep(0, self.t2n, self.id2n, open_stack=['XXX'])
        else:
            self.last_word_t = 0
            self.root = TreeNode(u'XXX', [], id=0, timestep=0)
            self.id2n[0] = self.root
            #self.root.set_timestep(0, self.t2n)
            # id = self.root.add_child(TreeNode(u'ROOT', []), self.id2n)
            # if id >= 0:
            #  self.open_nonterm_ids.append(id)
            #self.open_nonterm_ids.append(0)
    def reset_timestep(self):
        self.root.set_timestep(0, self.t2n, self.id2n, open_stack=['XXX'])

    def __str__(self):
        return self.root.to_parse_string()

    def to_parse_string(self):
        return self.root.to_parse_string()

    def copy(self):
        '''Return a deep copy of the current tree'''
        copied_tree = Tree()
        copied_tree.id2n = {}
        copied_tree.t2n = {}
        copied_tree.open_nonterm_ids = self.open_nonterm_ids[:]
        copied_tree.last_word_t = self.last_word_t

        root = TreeNode(u'trash', [])
        stack = [self.root]
        copy_stack = [root]
        while stack:
            cur = stack.pop()
            copy_cur = copy_stack.pop()
            copy_cur.label = cur.label
            copy_cur.children = []
            copy_cur.id = cur.id
            copy_cur.timestep = cur.timestep
            copy_cur.last_word_t = cur.last_word_t

            copied_tree.id2n[copy_cur.id] = copy_cur
            if copy_cur.timestep >= 0:
                copied_tree.t2n[copy_cur.timestep] = copy_cur
            for c in cur.children:
                if hasattr(c, 'set_parent'):
                    copy_c = TreeNode(c.label, [])
                    copy_cur.add_child(copy_c)

                    stack.append(c)
                    copy_stack.append(copy_c)
                else:
                    copy_cur.add_child(c)

        copied_tree.root = root
        return copied_tree

    @classmethod
    def from_rule_deriv(cls, derivs, wordswitch=True):
        tree = Tree()
        stack_tree = [tree.root]
        for x in derivs:
            r, stop = x
            p_tree = stack_tree.pop()
            if type(r) != Rule:
                if p_tree.label != u'*':
                    for i in derivs:
                        if type(i[0]) != Rule:
                            print i[0].encode('utf-8'), i[1]
                        else:
                            print i[0], i[1]
                assert p_tree.label == u'*', p_tree.label
                if wordswitch:
                    if r != Vocab.ES_STR:
                        p_tree.add_child(r)
                        stack_tree.append(p_tree)
                else:
                    p_tree.add_child(r)
                    if not stop:
                        stack_tree.append(p_tree)
                continue
            if p_tree.label == 'XXX':
                new_tree = TreeNode(r.lhs, [])
                p_tree.add_child(new_tree)
            else:
                if p_tree.label != r.lhs:
                    for i in derivs:
                        if type(i[0]) != Rule:
                            print i[0].encode('utf-8'), i[1]
                        else:
                            print i[0], i[1]
                    print tree.to_parse_string().encode('utf-8')
                    print p_tree.label.encode('utf-8'), r.lhs.encode('utf-8')
                    exit(1)
                assert p_tree.label == r.lhs, u"%s %s" % (p_tree.label, r.lhs)
                new_tree = p_tree
            open_nonterms = []
            for child in r.rhs:
                if child not in r.open_nonterms:
                    new_tree.add_child(child)
                else:
                    n = TreeNode(child, [])
                    new_tree.add_child(n)
                    open_nonterms.append(n)
            open_nonterms.reverse()
            stack_tree.extend(open_nonterms)
        return tree

    def to_string(self, piece=False):
        '''
    convert subtree into the sentence it represents
    '''
        return self.root.to_string(piece)

    def add_rule(self, id, rule):
        ''' Add one node to the tree based on current rule; only called on root tree '''
        node = self.id2n[id]
        node.set_timestep(len(self.t2n), self.t2n)
        node.last_word_t = self.last_word_t
        assert rule.lhs == node.label, "Rule lhs %s does not match the node %s to be expanded" % (rule.lhs, node.label)
        new_open_ids = []
        for rhs in rule.rhs:
            if rhs in rule.open_nonterms:
                new_open_ids.append(self.id2n[id].add_child(TreeNode(rhs, []), self.id2n))
            else:
                self.id2n[id].add_child(rhs)
                self.last_word_t = node.timestep
        new_open_ids.reverse()

        self.open_nonterm_ids.extend(new_open_ids)
        if self.open_nonterm_ids:
            node.frontir_label = self.id2n[self.open_nonterm_ids[-1]].label
        else:
            node.frontir_label = None

    def get_next_open_node(self):
        if len(self.open_nonterm_ids) == 0:
            print("stack empty, tree is complete")
            return -1
        return self.open_nonterm_ids.pop()

    def get_timestep_data(self, id):
        ''' Return a list of timesteps data associated with current tree node; only called on root tree '''
        data = []
        if self.id2n[id].parent():
            data.append(self.id2n[id].parent().timestep)
        else:
            data.append(0)
        data.append(self.id2n[id].last_word_t)
        return data

    def get_data_root(self, rule_vocab, word_vocab=None):
        data = []
        for t in xrange(1, len(self.t2n)):
            node = self.t2n[t]
            children, open_nonterms = [], []
            for c in node.children:
                if type(c) == str or type(c) == unicode:
                    children.append(c)
                else:
                    children.append(c.label)
                    open_nonterms.append(c.label)
            paren_t = 0 if not node.parent() else node.parent().timestep
            is_terminal = 1 if len(open_nonterms) == 0 else 0
            if word_vocab and is_terminal:
                leaf_len = len(node.children)
                for c in node.children:
                    d = [word_vocab.convert(c), paren_t, node.last_word_t, is_terminal, 0, leaf_len]
                    data.append(d)
                    leaf_len = -1
                data[-1][4] = 1
            else:
                d = [rule_vocab.convert(Rule(node.label, children, open_nonterms)), paren_t,
                    node.last_word_t, is_terminal,
                    rule_vocab.tag_vocab.convert(node.frontir_label), rule_vocab.tag_vocab.convert(node.label)]
                data.append(d)
        return data

    def get_bpe_rule(self, rule_vocab):
        ''' Get the rules for doing bpe. Label left and right child '''
        rule_idx = []
        for t in xrange(1, len(self.t2n)):
            node = self.t2n[t]
            children, open_nonterms = [], []
            child_idx = 1
            attach_tag = len(children) > 1
            for c in node.children:
                if type(c) == str or type(c) == unicode:
                    if attach_tag:
                        children.append(u'{}_{}'.format(c, child_idx))
                    else:
                        children.append(c)
                else:
                    if attach_tag:
                        children.append(u'{}_{}'.format(c.label, child_idx))
                    else:
                        children.append(c.label)
                    open_nonterms.append(c.label)
                child_idx += 1
            r = rule_vocab.convert(Rule(node.label, children, open_nonterms))
            rule_idx.append(r)
        return rule_idx

    def query_open_node_label(self):
        return self.id2n[self.open_nonterm_ids[-1]].label


def sent_piece_segs(p):
    '''
  Segment a sentence piece string into list of piece string for each word
  '''
    # print p
    # print p.split()
    # toks = re.compile(ur'\xe2\x96\x81[^(\xe2\x96\x81)]+')
    toks = re.compile(ur'\u2581')
    ret = []
    p_start = 0
    for m in toks.finditer(p):
        pos = m.start()
        if pos == 0:
            continue
        ret.append(p[p_start:pos])
        p_start = pos
    if p_start != len(p) - 1:
        ret.append(p[p_start:])
    return ret


def split_sent_piece(root, piece_l, word_idx):
    '''
  Split words into sentence piece
  '''
    new_children = []
    for i, c in enumerate(root.children):
        if type(c) == str or type(c) == unicode:
            # find number of empty space in c
            # space_c = len(c.decode('utf-8').split()) - 1
            # piece = []
            # for p in piece_l[word_idx:word_idx+space_c+1]:
            # piece.extend(p.split())
            # word_idx += (space_c + 1)
            #print c
            #print piece_l, word_idx
            piece = piece_l[word_idx].split()
            word_idx += 1
            # if u"".join(piece) != u'\u2581'+c and c != '-LRB-' and c != '-RRB-':
            #  print c.decode('utf-8').split()
            #  print piece
            new_children.extend(piece)
            #if len(piece) == 1:
                #n = TreeNode(u'x', piece)
                #n._parent = root
                #new_children.append(n)
                #new_children.append(piece[0])
                #continue
            #for p in piece:
                #n = TreeNode(u'x', [p])
                #r = TreeNode(root.label + u"_sub", [n])
                #r._parent = root
                #r = TreeNode(root.label, [p])
                #r._parent = root
                #new_children.append(r)
        else:
            word_idx = split_sent_piece(c, piece_l, word_idx)
            new_children.append(c)
    root.children = new_children
    return word_idx


def right_binarize(root, read_word=False):
    '''
  Right binarize a CusTree object
  read_word: if true, do not binarize terminal rules
  '''
    if type(root) == str or type(root) == unicode:
        return root
    if read_word and root.label == u'*':
        return root
    if len(root.children) <= 2:
        new_children = []
        for c in root.children:
            new_children.append(right_binarize(c))
        root.children = new_children
    else:
        if "__" in root.label:
            new_label = root.label
        else:
            new_label = root.label + "__"
        n_left_child = TreeNode(new_label, root.children[1:])
        n_left_child._parent = root
        root.children = [right_binarize(root.children[0]), right_binarize(n_left_child)]
    return root

def add_preterminal(root):
    ''' Add preterminal X before each terminal symbol '''

    for i, c in enumerate(root.children):
        if type(c) == str or type(c) == unicode:
            n = TreeNode(u'*', [c])
            n.set_parent(root)
            root.children[i] = n
        else:
            add_preterminal(c)

def add_preterminal_wordswitch(root, add_eos):
    ''' Add preterminal X before each terminal symbol '''
    ''' word_switch: one * symbol for each phrase chunk
        preterm_paren: * preterm parent already created
    '''
    preterm_paren = None
    new_children = []
    if root.label == u'*':
        if add_eos:
            root.add_child(Vocab.ES_STR)
        return root
    for i, c in enumerate(root.children):
        if type(c) == str or type(c) == unicode:
            if not preterm_paren:
                preterm_paren = TreeNode(u'*', [])
                preterm_paren.set_parent(root)
                new_children.append(preterm_paren)
            preterm_paren.add_child(c)
        else:
            if preterm_paren and add_eos:
                preterm_paren.add_child(Vocab.ES_STR)
            c = add_preterminal_wordswitch(c, add_eos)
            new_children.append(c)
            preterm_paren = None
    if preterm_paren and add_eos:
        preterm_paren.add_child(Vocab.ES_STR)
    root.children = new_children
    return root

def remove_preterminal_POS(root):
    ''' Remove the POS tag before terminal '''
    for i, c in enumerate(root.children):
        if c.is_preterminal():
            root.children[i] = c.children[0]
        else:
            remove_preterminal_POS(c)

def merge_depth(root, max_depth, cur_depth):
    ''' raise up trees whose depth exceed max_depth '''
    if cur_depth >= max_depth:
        #root.label = u'*'
        root.children = root.leaf_nodes()
        return root
    new_children = []
    for i, c in enumerate(root.children):
        if hasattr(c, 'children'):
            c = merge_depth(c, max_depth, cur_depth+1)
            # combine consecutive * nodes
            if new_children and hasattr(new_children[-1], 'label') and new_children[-1].is_preterminal() and c.is_preterminal():
                for x in c.children:
                    new_children[-1].add_child(x)
            else:
                new_children.append(c)
        else:
            new_children.append(c)
    root.children = new_children
    return root

def merge_tags(root):
    ''' raise up trees whose label is in a given set '''
    kept_label = set([u'np', u'vp', u'pp', u's', u'root', u'sbar', u'sinv', u'XXX', u'prn', u'adjp', u'advp',
                        u'whnp', u'whadvp',
                      u'NP', u'VP', u'PP',u'S', u'ROOT', u'SBAR', u'FRAG', u'SINV', u'PRN'])
    if not root.label in kept_label:
        root.label = u'xx'
        #root.children = root.leaf_nodes()
        #return root
    #new_children = []
    for i, c in enumerate(root.children):
        if hasattr(c, 'children'):
            c = merge_tags(c)
            # combine consecutive * nodes
            #if new_children and hasattr(new_children[-1], 'label') and new_children[-1].is_preterminal() and c.is_preterminal():
            #    for x in c.children:
            #        new_children[-1].add_child(x)
            #else:
            #    new_children.append(c)
        #else:
            #new_children.append(c)
        root.children[i] = c
    #root.children = new_children
    return root

def combine_tags(root):
    tag_dict = {'adjp': 'advp', 'sq': 'sbarq', 'whadjp': 'whadvp'}

# Tokenize a string.
# Tokens yielded are of the form (type, string)
# Possible values for 'type' are '(', ')' and 'WORD'
def tokenize(s):
    toks = re.compile(ur' +|[^() ]+|[()]')
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
        # print ty, s
        if ty == '(':
            children.append(parse_inner(toks))
        elif ty == ')':
            return TreeNode(name, children)
        else:
            children.append(s)


class ParseError(Exception):
    pass


# Parse this grammar:
# ROOT ::= '(' INNER
# INNER ::= WORD ROOT* ')'
# WORD ::= [A-Za-z]+
def parse_root(toks):
    ty, s = next(toks)
    if ty != '(':
        # print ty, s
        raise ParseError
    return parse_inner(toks)


if __name__ == "__main__":
    # test on copy
    '''
    s = "(ROOT (S (NP (FW i)) (VP (VBP like) (NP (PRP$ my) (NN steak) (NN medium))) (. .)) )"
    tree = Tree(parse_root(tokenize(s)), binarize=True)

    print tree
    r = RuleVocab()
    data = tree.get_data_root(r)
    for d in data:
        print d
        print r[d[0]], "frontir: ", r.tag_vocab[d[5]]
    for i, n in tree.id2n.items():
        print i, str(n), n.last_word_t, n.sib_t

    cop = tree.copy()
    print cop
    for i, n in cop.id2n.items():
        print i, n.to_parse_string(), n.last_word_t

    # test on construction
    tree = Tree()
    tree.add_rule(tree.get_next_open_node(), Rule('ROOT', ['S'], ['S']))
    tree.add_rule(tree.get_next_open_node(), Rule('S', ['NP', 'VP', '.'], ['NP', 'VP', '.']))
    tree.add_rule(tree.get_next_open_node(), Rule('NP', ['NNP'], ['NNP']))
    copied = tree.copy()
    tree.add_rule(tree.get_next_open_node(), Rule('NNP', ['I'], []))
    tree.add_rule(tree.get_next_open_node(), Rule('VP', ['am'], []))
    tree.add_rule(tree.get_next_open_node(), Rule('.', ['.'], []))
    # tree = right_binarize(tree)
    print tree
    print str(tree.open_nonterm_ids)
    for i, n in tree.id2n.items():
        print i, n.to_parse_string(), n.last_word_t
    for i, n in tree.t2n.items():
        print i, n.to_parse_string(), n.last_word_t
    print tree.get_data_root(RuleVocab())

    copied.add_rule(copied.get_next_open_node(), Rule('NNP', ['she'], []))
    copied.add_rule(copied.get_next_open_node(), Rule('VP', ['is'], []))
    copied.add_rule(copied.get_next_open_node(), Rule('.', ['.'], []))

    print
    print copied.to_parse_string()
    print str(copied.open_nonterm_ids)
    for i, n in copied.id2n.items():
        print i, n.to_parse_string(), copied.get_timestep_data(i)
    for i, n in copied.t2n.items():
        print i, n.to_parse_string(), n.last_word_t

    print
    '''

    '''
    rules = [Rule('ROOT', ['S'], ['S']), Rule('S', ['NP', 'VP', '.'], ['NP', 'VP', '.']), Rule('NP', ['NNP'], ['NNP']), \
             Rule('NNP', ['I'], []), Rule('VP', ['am'], []), Rule('.', ['.'], [])]
    tree = Tree.from_rule_deriv(rules)
    print tree
    print tree.to_string()
    for i, n in tree.id2n.items():
        print i, str(n)
    for i, n in tree.t2n.items():
        print i, str(n)

    r = Rule('ROOT', ['S'], ['S'])
    print isinstance(r, Rule)
    print u"{}\n".format(r)
    '''


    #train_parse = "/Users/cindywang/Documents/research/TSG/xnmt/orm_data/set0-trainunfilt.tok.parse.eng"
    #train_piece = "/Users/cindywang/Documents/research/TSG/xnmt/orm_data/set0-trainunfilt.tok.piece.eng"
    train_parse = "/Users/cindywang/Documents/research/TSG/xnmt/kftt_data/tok/kyoto-train.lowparse.en"
    train_piece = "/Users/cindywang/Documents/research/TSG/xnmt/kftt_data/tok/kyoto-train.lowpiece.en"
    parse_fp = codecs.open(train_parse, 'r', encoding='utf-8')
    piece_fp = codecs.open(train_piece, 'r', encoding='utf-8')
    rule_vocab = RuleVocab()
    word_vocab = Vocab()
    '''
    leaf_lens = defaultdict(int)
    for parse, piece in zip(parse_fp, piece_fp):
        t = Tree(parse_root(tokenize(parse)))
        remove_preterminal_POS(t.root)
        #t.root = right_binarize(t.root)
        split_sent_piece(t.root, sent_piece_segs(piece), 0)
        #merge_depth(t.root, 5, 0)
        merge_tags(t.root)
        add_preterminal_wordswitch(t.root, add_eos=False)
        t.reset_timestep()
        t.root.get_leaf_lens(leaf_lens)
        #t.get_data_root(rule_vocab)
        #print t.to_parse_string().encode('utf-8')
    for w in sorted(leaf_lens, key=leaf_lens.get, reverse=True):
        print 'len', w, 'count', leaf_lens[w]
    '''
    s = u"(root (s (np (fw i)) (vp (vbp like) (np (prp$ my) (nn steak) (nn medium))) (. .)) )"
    piece = u"\u2581i \u2581like \u2581my \u2581st eak \u2581medium \u2581."
    #s = u"(ROOT (S (NP (FW i)) (VP (VBD heard) (SBAR (IN that) (S (NP (PRP he)) (VP (VBD gave) (NP (PRP himself)) (PRT (RP up)) (PP (TO to) (NP (DT the) (NN police))))))) (. .)) )"
    tree = Tree(parse_root(tokenize(s)))
    # 1. remove pos tag 2. split sentence piece, 3. right binarize 4. add x preterminal
    remove_preterminal_POS(tree.root)
    split_sent_piece(tree.root, sent_piece_segs(piece), 0)
    #merge_depth(tree.root, 3, 0)
    merge_tags(tree.root)
    tree.root = add_preterminal_wordswitch(tree.root, add_eos=False)
    #tree.root = right_binarize(tree.root)
    
    tree.reset_timestep()
    data = tree.get_data_root(rule_vocab, word_vocab)
    for d in data:
        print d
        if d[3] == 0:
            print rule_vocab[d[0]]
        else:
            print word_vocab[d[0]]
    print tree.to_parse_string().encode('utf-8')
    print tree.to_string().encode('utf-8')
    #idx_list = rule_vocab.rule_index_with_lhs('*')
    #print len(idx_list)
    #print rule_vocab[idx_list[0]]
    print 'vocab size: ', len(rule_vocab)


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

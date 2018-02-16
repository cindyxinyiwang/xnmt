from __future__ import division, generators

import six
import plot
import dynet as dy
import numpy as np

import xnmt.length_normalization
import xnmt.batcher

from xnmt.vocab import Vocab
from xnmt.events import register_xnmt_event_assign, register_handler
from xnmt.generator import GeneratorModel
from xnmt.serializer import Serializable, DependentInitParam
from xnmt.search_strategy import BeamSearch, GreedySearch, Sampling
from xnmt.training_strategy import *
from xnmt.output import TextOutput, TreeHierOutput
from xnmt.reports import Reportable
import xnmt.serializer
from xnmt.batcher import mark_as_batch, is_batched
from xnmt.loss import LossBuilder
from xnmt.decoder import TreeDecoder, TreeHierDecoder, TreeHierFixtransDecoder
from xnmt.sentence_stats import *
from xnmt.length_normalization import *
# Reporting purposes
from lxml import etree

class Translator(GeneratorModel):
  '''
  A template class implementing an end-to-end translator that can calculate a
  loss and generate translations.
  '''

  def calc_loss(self, src, trg):
    '''Calculate loss based on input-output pairs.

    :param src: The source, a sentence or a batch of sentences.
    :param trg: The target, a sentence or a batch of sentences.
    :returns: An expression representing the loss.
    '''
    raise NotImplementedError('calc_loss must be implemented for Translator subclasses')

  def set_trg_vocab(self, trg_vocab=None):
    """
    Set target vocab for generating outputs.

    :param trg_vocab: target vocab, or None to generate word ids
    """
    self.trg_vocab = trg_vocab

  def set_post_processor(self, post_processor, sampling=False, output_beam=0):
    self.post_processor = post_processor
    self.sampling = sampling
    self.output_beam = output_beam

class DefaultTranslator(Translator, Serializable, Reportable):
  '''
  A default translator based on attentional sequence-to-sequence models.
  '''

  yaml_tag = u'!DefaultTranslator'

  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder, loop_trg=False, tag_embedder=None, word_attender=None, word_embedder=None):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    register_handler(self)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.loop_trg = loop_trg
    self.tag_embedder = tag_embedder
    self.word_attender = word_attender
    self.word_embedder = word_embedder
    if isinstance(decoder, TreeDecoder) and decoder.set_word_lstm:
      assert word_attender
    if isinstance(decoder, TreeHierDecoder):
      assert word_embedder
      assert word_attender

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]

  def dependent_init_params(self):
    ret = [DependentInitParam(param_descr="src_embedder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab_size()),
            DependentInitParam(param_descr="decoder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="trg_embedder.vocab_size", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
            DependentInitParam(param_descr="src_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab),
            DependentInitParam(param_descr="trg_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab)]
    if hasattr(self, 'word_embedder'):
      ret += [DependentInitParam(param_descr="word_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab.word_vocab),
              DependentInitParam(param_descr="word_embedder.vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.word_vocab_size()),
              DependentInitParam(param_descr="decoder.word_vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.word_vocab_size())]
    if hasattr(self, 'tag_embedder'):
      ret += [DependentInitParam(param_descr="tag_embedder.vocab", value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab.tag_vocab),
              DependentInitParam(param_descr="tag_embedder.vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.tag_vocab_size())]
    return ret


  def initialize_generator(self, train_src, train_trg, **kwargs):
    if kwargs.get("len_norm_type", None) is None:
      len_norm = xnmt.length_normalization.NoNormalization()
    else:
      if type(kwargs["len_norm_type"]) == MultinomialNormalization:
        len_norm_args = kwargs["len_norm_type"]
        sent_stats = SentenceStats()
        sent_stats.populate_statistics(train_src, train_trg)
        len_norm = MultinomialNormalization(sent_stats, m=len_norm_args.m,
                                            apply_during_search=len_norm_args.apply_during_search)
      elif type(kwargs["len_norm_type"]) == GaussianNormalization:
        len_norm_args = kwargs["len_norm_type"]
        sent_stats = SentenceStats()
        sent_stats.populate_statistics(train_src, train_trg)
        len_norm = GaussianNormalization(sent_stats,
                                            apply_during_search=len_norm_args.apply_during_search,
                                            length_ratio=len_norm_args.length_ratio,
                                            div=len_norm_args.div)
      else:
        len_norm = xnmt.serializer.YamlSerializer().initialize_object(kwargs["len_norm_type"])
    search_args = {}
    if kwargs.get("max_len", None) is not None: search_args["max_len"] = kwargs["max_len"]
    self.sample_num = kwargs["sample_num"]
    self.sampling = False
    self.output_beam = False
    if kwargs.get("sample_num", -1) > 0:
      search_args["sample_num"] = kwargs["sample_num"]
      self.search_strategy = Sampling(**search_args)
      self.sampling= True
    else:
      if kwargs.get("beam", None) is None:
        self.search_strategy = GreedySearch(**search_args)
      else:
        search_args["beam_size"] = kwargs.get("beam", 1)
        search_args["len_norm"] = len_norm
        print(type(len_norm))
        self.search_strategy = BeamSearch(**search_args)
        if kwargs.get("output_beam", 0) > 0:
          self.output_beam = kwargs["output_beam"]
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def initialize_training_strategy(self, training_strategy):
    self.loss_calculator = training_strategy

  def calc_loss(self, src, trg, trg_rule_vocab=None, word_vocab=None):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded); losses will be accumulated only if trg_mask[batch,pos]==0, or no mask is set
    :returns: (possibly batched) loss expression
    """
    assert hasattr(self, "loss_calculator")
    if hasattr(self.decoder, 'decoding'):
      self.decoder.decoding = False
    # Initialize the hidden state from the encoder
    if self.loop_trg:
      loss = []
      
      ss = mark_as_batch([Vocab.SS])
      emb_ss = self.trg_embedder.embed(ss)
      #self.start_sent()
      #embeddings = self.src_embedder.embed_sent(src)
      #encodings = self.encoder(embeddings)
      #self.attender.init_sent(encodings)
      #enc_final_state = self.encoder.get_final_states()
      for i in xrange(len(trg)):
        self.start_sent()
        embeddings = self.src_embedder.embed_sent(src[i])
        encodings = self.encoder(embeddings)
        self.attender.init_sent(encodings)
        if self.word_attender:
          self.word_attender.init_sent(encodings)
        if trg.mask:
          single_trg = mark_as_batch([trg[i]], xnmt.batcher.Mask(np.expand_dims(trg.mask.np_arr[i], 0)))
        else:
          single_trg = mark_as_batch([trg[i]])
        #dec_state = self.decoder.initial_state([final_state.pick_batch_elem(i) for final_state in enc_final_state], emb_ss)
        dec_state = self.decoder.initial_state(self.encoder.get_final_states(), emb_ss)
        #loss.append(self.loss_calculator(self, dec_state, mark_as_batch(src[i]), single_trg, pick_src_elem=i, trg_rule_vocab=trg_rule_vocab))
        loss.append(self.loss_calculator(self, dec_state, mark_as_batch(src[i]), single_trg,
                                         trg_rule_vocab=trg_rule_vocab, word_vocab=word_vocab))
        #dy.print_text_graphviz()
        #exit(0)
      return dy.esum(loss)
    else:
      self.start_sent()
      embeddings = self.src_embedder.embed_sent(src)
      encodings = self.encoder(embeddings)
      self.attender.init_sent(encodings)
      if self.word_attender:
        self.word_attender.init_sent(encodings)
      ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
      dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
      return self.loss_calculator(self, dec_state, src, trg, trg_rule_vocab=trg_rule_vocab)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None, trg_rule_vocab=None, word_vocab=None):
    if hasattr(self.decoder, 'decoding'):
      self.decoder.decoding=True
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    scores = []
    for sents in src:
      if self.sample_num > 0:
        output_actions = []
        score = []
        for i in range(self.sample_num):
          self.start_sent()
          embeddings = self.src_embedder.embed_sent(src)
          encodings = self.encoder(embeddings)
          self.attender.init_sent(encodings)
          if self.word_attender:
            self.word_attender.init_sent(encodings)
          ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
          if isinstance(self.decoder, TreeDecoder) or isinstance(self.decoder, TreeHierDecoder) or isinstance(self.decoder, TreeHierFixtransDecoder):

            dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss),
                                                   decoding=True)
          else:
            dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
          o, s = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder,
                                                                       dec_state, src_length=len(sents),
                                                                       forced_trg_ids=forced_trg_ids,
                                                                       trg_rule_vocab=trg_rule_vocab,
                                                                       tag_embedder=self.tag_embedder,
                                                                       word_attender=self.word_attender,
                                                                       word_embedder=self.word_embedder)
          output_actions.append(o)
          score.append(s)
          dy.renew_cg()
      else:
        self.start_sent()
        embeddings = self.src_embedder.embed_sent(src)
        encodings = self.encoder(embeddings)
        self.attender.init_sent(encodings)
        if self.word_attender:
          self.word_attender.init_sent(encodings)
        ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
        if isinstance(self.decoder, TreeDecoder) or isinstance(self.decoder, TreeHierDecoder) or isinstance(self.decoder, TreeHierFixtransDecoder):
          dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss),
                                                 decoding=True)
        else:
          dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss))
        output_actions, score = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder,
                                                                     dec_state, src_length=len(sents),
                                                                     forced_trg_ids=forced_trg_ids, trg_rule_vocab=trg_rule_vocab, tag_embedder=self.tag_embedder,
                                                                     word_attender=self.word_attender, word_embedder=self.word_embedder,
                                                                     output_beam=self.output_beam)
      # In case of reporting
      if self.report_path is not None:
        src_words = [self.reporting_src_vocab[w] for w in sents]
        trg_words = [self.trg_vocab[w] for w in output_actions[1:]]
        attentions = self.attender.attention_vecs
        self.set_report_input(idx, src_words, trg_words, attentions)
        self.set_report_resource("src_words", src_words)
        self.set_report_path('{}.{}'.format(self.report_path, str(idx)))
        self.generate_report(self.report_type)
      # Append output to the outputs
      if self.sampling or self.output_beam:
        if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
          if self.word_embedder:
            for action in output_actions:
              outputs.append( TreeHierOutput(action, rule_vocab=self.trg_vocab, word_vocab=word_vocab) )
          else:
            for action in output_actions:
              outputs.append(TextOutput(action, self.trg_vocab))
        else:
          for action in output_actions:
            outputs.append(action)
        scores.extend(score)
      else:
        if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
          if self.word_embedder:
            outputs.append(TreeHierOutput(output_actions, rule_vocab=self.trg_vocab, word_vocab=word_vocab))
          else:
            outputs.append(TextOutput(output_actions, self.trg_vocab))
        else:
          outputs.append((output_actions, score))
    if self.sampling or self.output_beam:
      return outputs, scores
    else:
      return outputs

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab

  @register_xnmt_event_assign
  def html_report(self, context=None):
    assert(context is None)
    idx, src, trg, att = self.get_report_input()
    path_to_report = self.get_report_path()
    html = etree.Element('html')
    head = etree.SubElement(html, 'head')
    title = etree.SubElement(head, 'title')
    body = etree.SubElement(html, 'body')
    report = etree.SubElement(body, 'h1')
    if idx is not None:
      title.text = report.text = 'Translation Report for Sentence %d' % (idx)
    else:
      title.text = report.text = 'Translation Report'
    main_content = etree.SubElement(body, 'div', name='main_content')

    # Generating main content
    captions = [u"Source Words", u"Target Words"]
    inputs = [src, trg]
    for caption, inp in six.moves.zip(captions, inputs):
      if inp is None: continue
      sent = ' '.join(inp)
      p = etree.SubElement(main_content, 'p')
      p.text = u"{}: {}".format(caption, sent)

    # Generating attention
    if not any([src is None, trg is None, att is None]):
      attention = etree.SubElement(main_content, 'p')
      att_text = etree.SubElement(attention, 'b')
      att_text.text = "Attention:"
      etree.SubElement(attention, 'br')
      attention_file = u"{}.attention.png".format(path_to_report)

      if type(att) == dy.Expression:
        attentions = att.npvalue()
      elif type(att) == list:
        attentions = np.concatenate([x.npvalue() for x in att], axis=1)
      elif type(att) != np.ndarray:
        raise RuntimeError("Illegal type for attentions in translator report: {}".format(type(attentions)))
      plot.plot_attention(src, trg, attentions, file_name = attention_file)

    # return the parent context to be used as child context
    return html


class TreeTranslator(Translator, Serializable, Reportable):
  '''
  A default translator based on attentional sequence-to-sequence models.
  '''

  yaml_tag = u'!TreeTranslator'

  def __init__(self, src_embedder, encoder, attender, trg_embedder, decoder,
               word_attender=None, word_embedder=None):
    '''Constructor.

    :param src_embedder: A word embedder for the input language
    :param encoder: An encoder to generate encoded inputs
    :param attender: An attention module
    :param trg_embedder: A word embedder for the output language
    :param decoder: A decoder
    '''
    register_handler(self)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.trg_embedder = trg_embedder
    self.decoder = decoder
    self.word_attender = word_attender
    self.word_embedder = word_embedder
    if isinstance(decoder, TreeDecoder) and decoder.set_word_lstm:
      assert word_attender
    if isinstance(decoder, TreeHierDecoder):
      assert word_embedder
      assert word_attender

  def shared_params(self):
    return [set(["src_embedder.emb_dim", "encoder.input_dim"]),
            set(["encoder.hidden_dim", "attender.input_dim", "decoder.input_dim"]),
            set(["attender.state_dim", "decoder.lstm_dim"]),
            set(["trg_embedder.emb_dim", "decoder.trg_embed_dim"])]

  def dependent_init_params(self):
    ret = [DependentInitParam(param_descr="src_embedder.vocab_size",
                              value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab_size()),
           DependentInitParam(param_descr="decoder.vocab_size",
                              value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
           DependentInitParam(param_descr="trg_embedder.vocab_size",
                              value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab_size()),
           DependentInitParam(param_descr="src_embedder.vocab",
                              value_fct=lambda: self.yaml_context.corpus_parser.src_reader.vocab),
           DependentInitParam(param_descr="trg_embedder.vocab",
                              value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab)]
    if hasattr(self, 'word_embedder'):
      ret += [DependentInitParam(param_descr="word_embedder.vocab",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab.word_vocab),
              DependentInitParam(param_descr="word_embedder.vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.word_vocab_size()),
              DependentInitParam(param_descr="decoder.word_vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.word_vocab_size())]
    if hasattr(self, 'tag_embedder'):
      ret += [DependentInitParam(param_descr="tag_embedder.vocab",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.vocab.tag_vocab),
              DependentInitParam(param_descr="tag_embedder.vocab_size",
                                 value_fct=lambda: self.yaml_context.corpus_parser.trg_reader.tag_vocab_size())]
    return ret

  def initialize_generator(self, train_src, train_trg, **kwargs):
    if kwargs.get("len_norm_type", None) is None:
      len_norm = xnmt.length_normalization.NoNormalization()
    else:
      if type(kwargs["len_norm_type"]) == MultinomialNormalization:
        len_norm_args = kwargs["len_norm_type"]
        sent_stats = SentenceStats()
        sent_stats.populate_statistics(train_src, train_trg)
        len_norm = MultinomialNormalization(sent_stats, m=len_norm_args.m,
                                            apply_during_search=len_norm_args.apply_during_search)
      elif type(kwargs["len_norm_type"]) == GaussianNormalization:
        len_norm_args = kwargs["len_norm_type"]
        sent_stats = SentenceStats()
        sent_stats.populate_statistics(train_src, train_trg)
        len_norm = GaussianNormalization(sent_stats,
                                            apply_during_search=len_norm_args.apply_during_search,
                                            length_ratio=len_norm_args.length_ratio,
                                            div=len_norm_args.div)
      else:
        len_norm = xnmt.serializer.YamlSerializer().initialize_object(kwargs["len_norm_type"])
    search_args = {}
    if kwargs.get("max_len", None) is not None: search_args["max_len"] = kwargs["max_len"]
    self.sample_num = kwargs["sample_num"]
    self.sampling = False
    self.output_beam = False
    if kwargs.get("sample_num", -1) > 0:
      search_args["sample_num"] = kwargs["sample_num"]
      self.search_strategy = Sampling(**search_args)
      self.sampling = True
    else:
      if kwargs.get("beam", None) is None:
        self.search_strategy = GreedySearch(**search_args)
      else:
        search_args["beam_size"] = kwargs.get("beam", 1)
        search_args["len_norm"] = len_norm
        self.search_strategy = BeamSearch(**search_args)
        if kwargs.get("output_beam", 0) > 0:
          self.output_beam = kwargs["output_beam"]
    self.report_path = kwargs.get("report_path", None)
    self.report_type = kwargs.get("report_type", None)

  def initialize_training_strategy(self, training_strategy):
    assert type(training_strategy) == TrainingTreeLoss
    self.loss_calculator = training_strategy

  def calc_loss(self, src, trg, trg_rule_vocab=None, word_vocab=None):
    """
    :param src: source sequence (unbatched, or batched + padded)
    :param trg: target sequence (unbatched, or batched + padded); losses will be accumulated only if trg_mask[batch,pos]==0, or no mask is set
    :returns: (possibly batched) loss expression
    """
    assert hasattr(self, "loss_calculator")
    if hasattr(self.decoder, 'decoding'):
      self.decoder.decoding = False
    # Initialize the hidden state from the encoder

    rule_losses = []
    word_losses = []
    word_eos_losses = []
    rule_count, word_count, word_eos_count = 0, 0, 0
    ss = mark_as_batch([Vocab.SS])
    emb_ss = self.trg_embedder.embed(ss)
    # self.start_sent()
    # embeddings = self.src_embedder.embed_sent(src)
    # encodings = self.encoder(embeddings)
    # self.attender.init_sent(encodings)
    # enc_final_state = self.encoder.get_final_states()
    for i in xrange(len(trg)):
      self.start_sent()
      embeddings = self.src_embedder.embed_sent(src[i])
      encodings = self.encoder(embeddings)
      self.attender.init_sent(encodings)
      self.word_attender.init_sent(encodings)
      if trg.mask:
        single_trg = mark_as_batch([trg[i]], xnmt.batcher.Mask(np.expand_dims(trg.mask.np_arr[i], 0)))
      else:
        single_trg = mark_as_batch([trg[i]])
      dec_state = self.decoder.initial_state(self.encoder.get_final_states(), emb_ss)
      rule_loss, word_loss, word_eos_loss, rule_c, word_c, word_eos_c \
        = self.loss_calculator(self, dec_state, mark_as_batch(src[i]), single_trg, trg_rule_vocab=trg_rule_vocab, word_vocab=word_vocab)
      rule_losses.append(rule_loss)
      word_losses.append(word_loss)
      word_eos_losses.append(word_eos_loss)
      rule_count += rule_c
      word_count += word_c
      word_eos_count += word_eos_c
      # dy.print_text_graphviz()
      # exit(0)
    return (dy.esum(rule_losses), dy.esum(word_losses), dy.esum(word_eos_losses), rule_count, word_count, word_eos_count)

  def generate(self, src, idx, src_mask=None, forced_trg_ids=None, trg_rule_vocab=None, word_vocab=None):
    if hasattr(self.decoder, 'decoding'):
      self.decoder.decoding = True
    if not xnmt.batcher.is_batched(src):
      src = xnmt.batcher.mark_as_batch([src])
    else:
      assert src_mask is not None
    outputs = []
    scores = []
    for sents in src:
      if self.sample_num > 0:
        output_actions = []
        score = []
        for i in range(self.sample_num):
          self.start_sent()
          embeddings = self.src_embedder.embed_sent(src)
          encodings = self.encoder(embeddings)
          self.attender.init_sent(encodings)

          self.word_attender.init_sent(encodings)
          ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS

          dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss),
                                                   decoding=True)
          o, s = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder,
                                                      dec_state, src_length=len(sents),
                                                      forced_trg_ids=forced_trg_ids,
                                                      trg_rule_vocab=trg_rule_vocab,
                                                      tag_embedder=self.tag_embedder,
                                                      word_attender=self.word_attender,
                                                      word_embedder=self.word_embedder)
          output_actions.append(o)
          score.append(s)
          dy.renew_cg()
      else:
        self.start_sent()
        embeddings = self.src_embedder.embed_sent(src)
        encodings = self.encoder(embeddings)
        self.attender.init_sent(encodings)
        if self.word_attender:
          self.word_attender.init_sent(encodings)
        ss = mark_as_batch([Vocab.SS] * len(src)) if is_batched(src) else Vocab.SS
        dec_state = self.decoder.initial_state(self.encoder.get_final_states(), self.trg_embedder.embed(ss),
                                                decoding=True)
        output_actions, score = self.search_strategy.generate_output(self.decoder, self.attender, self.trg_embedder,
                                                                     dec_state, src_length=len(sents),
                                                                     forced_trg_ids=forced_trg_ids,
                                                                     trg_rule_vocab=trg_rule_vocab,
                                                                     word_attender=self.word_attender,
                                                                     word_embedder=self.word_embedder,
                                                                     output_beam=self.output_beam)

      # Append output to the outputs
      if self.sampling or self.output_beam:
        if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
          if self.word_embedder:
            for action in output_actions:
              outputs.append(TreeHierOutput(action, rule_vocab=self.trg_vocab, word_vocab=word_vocab))
          else:
            for action in output_actions:
              outputs.append(TextOutput(action, self.trg_vocab))
        else:
          for action in output_actions:
            outputs.append(action)
        scores.extend(score)
      else:
        if hasattr(self, "trg_vocab") and self.trg_vocab is not None:
          if self.word_embedder:
            outputs.append(TreeHierOutput(output_actions, rule_vocab=self.trg_vocab, word_vocab=word_vocab))
          else:
            outputs.append(TextOutput(output_actions, self.trg_vocab))
        else:
          outputs.append((output_actions, score))
    if self.sampling or self.output_beam:
      return outputs, scores
    else:
      return outputs

  def set_reporting_src_vocab(self, src_vocab):
    """
    Sets source vocab for reporting purposes.
    """
    self.reporting_src_vocab = src_vocab
# coding: utf-8

import io
import sys

import dynet as dy

from xnmt.output import *
from xnmt.serializer import *
from xnmt.retriever import *
from xnmt.translator import *
from xnmt.search_strategy import *
from xnmt.options import OptionParser, Option


'''
This will be the main class to perform decoding.
'''

options = [
  Option("dynet-mem", int, required=False),
  Option("dynet-gpu-ids", int, required=False),
  Option("dynet-gpus", int, required=False),
  Option("model_file", force_flag=True, required=True, help_str="pretrained (saved) model path"),
  Option("src_file", help_str="path of input src file to be translated"),
  Option("trg_file", help_str="path of file where trg translatons will be written"),
  Option("ref_file", str, required=False, help_str="path of file with reference translations, e.g. for forced decoding"),
  Option("max_src_len", int, required=False, help_str="Remove sentences from data to decode that are longer than this on the source side"),
  Option("input_format", default_value="text", help_str="format of input data: text/contvec"),
  Option("post_process", default_value="none", help_str="post-processing of translation outputs: none/join-char/join-bpe/rule-piece/rule/ccg-piece"),
  Option("tag_file", default_value="none", help_str="tag file for ccg-piece"),
  Option("candidate_id_file", required=False, default_value=None, help_str="if we are doing something like retrieval where we select from fixed candidates, sometimes we want to limit our candidates to a certain subset of the full set. this setting allows us to do this."),
  Option("report_path", str, required=False, help_str="a path to which decoding reports will be written"),
  Option("report_type", str, default_value="html", required=False, help_str="report to generate file/html. Can be multiple, separate with comma."),
  Option("beam", int, default_value=1),
  Option("max_len", int, default_value=100),
  Option("sample_num", int, default_value=-1, required=False),
  Option("output_beam", int, default_value=0, required=False),
  Option("len_norm_type", str, required=False),
  Option("mode", str, default_value="onebest", help_str="type of decoding to perform. onebest: generate one best. forced: perform forced decoding."),
]

NO_DECODING_ATTEMPTED = u"@@NO_DECODING_ATTEMPTED@@"

def xnmt_decode(args, model_elements=None, train_src=None, train_trg=None):
  """
  :param model_elements: If None, the model will be loaded from args.model_file. If set, should
  equal (corpus_parser, generator).
  """
  if model_elements is None:
    raise RuntimeError("xnmt_decode with model_element=None needs to be updated to run with the new YamlSerializer")
#    TODO: These lines need to be updated / removed!
#    model = dy.Model()
#    model_serializer = JSONSerializer()
#    serialize_container = model_serializer.load_from_file(args.model_file, model)
#
#    src_vocab = Vocab(serialize_container.src_vocab)
#    trg_vocab = Vocab(serialize_container.trg_vocab)
#
#    generator = DefaultTranslator(serialize_container.src_embedder, serialize_container.encoder,
#                                  serialize_container.attender, serialize_container.trg_embedder,
#                                  serialize_container.decoder)
#
  else:
    corpus_parser, generator = model_elements

  is_reporting = issubclass(generator.__class__, Reportable) and args.report_path is not None
  # Corpus
  src_corpus = corpus_parser.src_reader.read_sents(args.src_file)
  # Get reference if it exists and is necessary
  if args.mode == "forced":
    if args.ref_file == None:
      raise RuntimeError("When performing {} decoding of mode, must specify reference file".format(args.mode))
    ref_corpus = list(corpus_parser.trg_reader.read_sents(args.ref_file))
  else:
    ref_corpus = None
  # Vocab
  src_vocab = corpus_parser.src_reader.vocab if hasattr(corpus_parser.src_reader, "vocab") else None
  trg_vocab = corpus_parser.trg_reader.vocab if hasattr(corpus_parser.trg_reader, "vocab") else None
  word_vocab = corpus_parser.trg_reader.word_vocab if hasattr(corpus_parser.trg_reader, "word_vocab") else None
  # Perform initialization
  generator.set_train(False)
  generator.initialize_generator(train_src, train_trg, **args.params_as_dict)
  sampling = args.sample_num >=0
  output_beam = args.output_beam
  # TODO: Structure it better. not only Translator can have post processes
  if issubclass(generator.__class__, Translator):

    generator.set_post_processor(output_processor_for_spec(args.post_process, tag_file=args.tag_file),
                                 sampling=sampling,
                                 output_beam=args.output_beam)
    generator.set_trg_vocab(trg_vocab)
    generator.set_reporting_src_vocab(src_vocab)

  if is_reporting:
    generator.set_report_resource("src_vocab", src_vocab)
    generator.set_report_resource("trg_vocab", trg_vocab)

  rule_decode = False
  if hasattr(corpus_parser.trg_reader.vocab, 'rule_index_with_lhs'):
    rule_decode = True
    trg_parse = io.open(args.trg_file + ".parse", 'wt', encoding='utf-8')
  # Perform generation of output
  with io.open(args.trg_file, 'wt', encoding='utf-8') as fp:  # Saving the translated output to a trg file
    for i, src in enumerate(src_corpus):
      # Do the decoding
      if args.max_src_len is not None and len(src) > args.max_src_len:
        outputs = NO_DECODING_ATTEMPTED
      else:
        dy.renew_cg()
        if rule_decode:
          # output both the parse trees and sentence
          outputs, scores = generator.generate_output(src, i, trg_rule_vocab=trg_vocab, word_vocab=word_vocab)
          if sampling or output_beam:
            x = 0
            for o, t in outputs:
              trg_parse.write(u"{}\n".format(t))
              fp.write(u"{} ||| {} ||| {} ||| ll={}\n".format(i, o, scores[x], scores[x]))
              x += 1
          else:
            outputs, tree = outputs[0], outputs[1]
            trg_parse.write(u"{}\n".format(tree))
            fp.write(u"{}\n".format(outputs))
        else:
          outputs, scores = generator.generate_output(src, i)
          ref_ids = ref_corpus[i] if ref_corpus != None else None
          # Printing to trg file
          if sampling or output_beam:
            for o, s in zip(outputs, scores):
              fp.write(u"{} ||| {} ||| {} ||| ll={}\n".format(i, o, s, s))
          else:
            fp.write(u"{}\n".format(outputs))
  if rule_decode:
    trg_parse.close()

def output_processor_for_spec(spec, tag_file=None):
  if spec == "none":
    return PlainTextOutputProcessor()
  elif spec == "join-char":
    return JoinedCharTextOutputProcessor()
  elif spec == "join-bpe":
    return JoinedBPETextOutputProcessor()
  elif spec == "join-piece":
    return JoinedPieceTextOutputProcessor()
  elif spec == "rule-piece-wordswitch":
    return RuleOutputProcessor(piece=True, wordswitch=True)
  elif spec == "rule-piece":
    return RuleOutputProcessor(piece=True, wordswitch=False)
  elif spec == "rule-bpe":
    return RuleBPEOutputProcessor(wordswitch=False)
  elif spec == "rule-wordswitch":
    return RuleOutputProcessor(piece=True, wordswitch=True)
  elif spec == "rule":
    return RuleOutputProcessor(piece=False, wordswitch=False)
  elif spec == "ccg-piece":
    assert tag_file
    tags = set()
    with codecs.open(tag_file, 'r', encoding="utf-8") as myfile:
      for line in myfile:
        tags.add(line.strip())
    return CcgPieceOutputProcessor(tag_set=tags)
  else:
    raise RuntimeError("Unknown postprocessing argument {}".format(spec))

if __name__ == "__main__":
  # Parse arguments
  parser = OptionParser()
  parser.add_task("decode", options)
  args = parser.args_from_command_line("decode", sys.argv[1:])
  # Load model
  xnmt_decode(args)


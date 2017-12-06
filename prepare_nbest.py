import codecs
import sys

def prepare(src_file, nbest_file, src_out, nbest_out):
    nbest = codecs.open(nbest_file, 'r', encoding='utf-8').readlines()
    src = codecs.open(src_file, 'r', encoding='utf-8').readlines()
    src_out_fp = codecs.open(src_out, 'w', encoding='utf-8')
    nbest_out_fp = codecs.open(nbest_out, 'w', encoding='utf-8')
    #print src[0].encode('utf-8')
    for l in nbest:
      toks = l.split(u' ||| ')
      idx = int(toks[0])
      src_out_fp.write(u'{}\n'.format(src[idx].strip()))
      nbest_out_fp.write(u'{}\n'.format(toks[1]))

    src_out_fp.close()
    nbest_out_fp.close()

def add_score(nbest_file, score_file, nbest_out):
    nbest = codecs.open(nbest_file, 'r', encoding='utf-8').readlines()
    score_lines = codecs.open(score_file, 'r', encoding='utf-8').readlines()
    nbest_new = codecs.open(nbest_out, 'w', encoding='utf-8')
    
    scores = [t.split()[0] for t in score_lines ]
    counts = [t.split()[1] for t in score_lines ]
    for i, l in enumerate(nbest):
      toks = l.split(u' ||| ')
      toks[2] = toks[2] + u" rerankScore=%s" % (scores[i]) + u" unkCount=%s" % (counts[i])
      nbest_new.write(u' ||| '.join(toks))
    nbest_new.close()

if __name__ == "__main__":
    #nbest = "/usr2/data/junjieh/LORELEI/results/cp4/orm_eng_lex_abo_tok_nospm_lm3_v6/decode_cdec_nbest/ExtractSection.devtest/out_nbest"
    nbest = sys.argv[1] 
    nbest_out = "orm-eng/data/nbest_sents"
    nbest_final = sys.argv[2]
    #nbest_final = "orm-eng/data/nbest_final"
    add_score(nbest, nbest_out+'.score', nbest_final)

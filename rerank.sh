#! /bin/bash

SRC="/usr2/home/xinyiw1/rerank/xnmt/orm-eng/data/unseq-setE.tok.orm"
NBEST="/usr2/data/junjieh/LORELEI/results/cp4/orm_eng_lex_abo_tok_nospm_lm3_v6/decode_cdec_nbest/ExtractSection.devtest/out_nbest"
NBEST_OUT="/usr2/home/xinyiw1/rerank/xnmt/nbest.final"
PIECE_MOD="/home/gneubig/exp2/lorelei/retest2017/xnmt/archive/001-sentpiece/orm.model"

# intermediate files
PREP_SRC='orm-eng/data/nbest_src'
PREP_NBEST='orm-eng/data/nbest_sents'
PIECE_SRC='orm-eng/data/nbest_src.piece'
PIECE_NBEST='orm-eng/data/nbest_sents.piece'

# prepare src nbest parallel corpus
python /usr2/home/xinyiw1/rerank/xnmt/make_nbest_src.py $SRC $NBEST $PREP_SRC $PREP_NBEST

# sentence piece
/home/gneubig/usr/local/sentencepiece/src/spm_encode --model=$PIECE_MOD --output_format=piece < $PREP_SRC > $PIECE_SRC
/home/gneubig/usr/local/sentencepiece/src/spm_encode --model=$PIECE_MOD --output_format=piece < $PREP_NBEST > $PIECE_NBEST


# get score
python /usr2/home/xinyiw1/rerank/xnmt/xnmt/xnmt_run_experiments.py /usr2/home/xinyiw1/rerank/xnmt/rerank.yaml

# combine to make new nbest list
python /usr2/home/xinyiw1/rerank/xnmt/prepare_nbest.py $NBEST $NBEST_OUT

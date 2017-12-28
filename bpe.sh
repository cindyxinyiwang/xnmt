#! /bin/bash

python xnmt/bpe_tree.py \
	--tree_file examples/data/dev-head5.parse.en \
	--piece_file examples/data/dev-head5.piece.en \
	--max_iter 2 \
	--out_file dev_short.parse.en1

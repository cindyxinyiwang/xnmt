#!/bin/bash

python setup.py install
python xnmt/xnmt_run_experiments.py examples/kftt_wordlstm.yaml \
		--dynet-gpu \
		--dynet-mem 10000 

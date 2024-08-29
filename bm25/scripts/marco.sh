#!/bin/bash

build_marco() {
   python bm25/main.py \
    --build_index \
    --force_rebuild \
    --tokenizer_name bert-base-uncased \
    --corpus_path data/msmarco/bert/corpus_ids.json \
    --index_path bm25/index/marco \
    --index_name array_index.h5py \
}

# Search MARCO
search_marco() {
   OUTPATH=bm25/retrieve_results/marco/m25_k15.tsv
   python bm25/main.py --do_retrieve \
    --tokenizer_name bert-base-uncased \
    --query_path data/msmarco/bert/dev.query.json \
    --index_path bm25/index/marco \
    --index_name array_index.h5py \
    --output_path $OUTPATH \

   python utils/eval.py \
    --gt-path data/msmarco/qrels.dev.tsv \
    --qrels-path $OUTPATH
}

build_marco
search_marco

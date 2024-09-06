#!/bin/bash

build_nq() {
   python bm25/main.py \
    --build_index \
    --force_rebuild \
    --do_tokenize \
    --tokenizer_name bert-base-uncased \
    --corpus_path data/nq/nq_doc.tsv \
    --index_path bm25/index/nq \
    --index_name array_index.bin 
}
# --corpus_path data/msmarco/bert/corpus_ids.json \

# Search MARCO
search_nq() {
   OUTPATH=bm25/retrieve_results/nq/m25_mod.tsv
   python bm25/main.py --do_retrieve \
    --tokenizer_name bert-base-uncased \
    --query_path data/nq/bert/nq_query_ids.json \
    --index_path bm25/index/nq \
    --index_name array_index.bin \
    --output_path $OUTPATH \

   python utils/eval.py \
    --gt-path data/nq/nq_gt.tsv \
    --qrels-path $OUTPATH
}

# --query_path data/msmarco/bert/dev.query.json \

# build_nq
search_nq

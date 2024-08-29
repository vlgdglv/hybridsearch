#!/bin/bash

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



# Search NQ
search_nq(){
   python bm25/main.py --do_retrieve \
      --do_tokenize \
      --tokenizer_name Luyu/co-condenser-wiki  \
      --query_path data/nq/nq_query.tsv \
      --index_path bm25/index/nq \
      --index_name array_index.h5py \
      --output_path bm25/retrieve_results/nq/m25.tsv
      # --query_path data/nq/bert/nq_query_ids.json \

   python utils/eval.py \
      --gt-path data/nq/nq_gt.tsv \
      --qrels-path bm25/retrieve_results/nq/bm25.tsv
}

search_marco
# search_nq
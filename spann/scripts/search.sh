#!/bin/bash

export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

# Search MARCO
search_marco() {
  cd spann/index
  OUT=msmarco_8_spann
  python search.py \
  --query_emb_path ../../data/msmarco/spann/marco_query.pt \
  --spann_index marco_8 \
  --output_path ../../spann/retrieve_results/marco/$OUT.tsv
  
  cd ../..
  python utils/eval.py \
  --gt-path data/msmarco/qrels.dev.tsv \
  --qrels-path spann/retrieve_results/marco/$OUT.tsv
}



# Search NQ
search_nq(){
    cd spann/index
    OUT=nq_8_spann
    python search.py\
      --query_emb_path ../../data/nq/spann/nq_query.pt \
      --spann_index nq_8 \
      --output_path ../../spann/retrieve_results/nq/$OUT.tsv

    cd ../..
    python utils/eval.py \
      --gt-path data/nq/nq_gt.tsv \
      --qrels-path spann/retrieve_results/nq/$OUT.tsv
}

search_marco
# search_nq
#!/bin/bash



# Search NQ
search_nq(){
    cd /datacosmos/User/baoht/onesparse2/hybridsearch/spann/index
    python search.py\
      --query_emb_path /datacosmos/User/baoht/onesparse2/OneSparse/data/nq/onesparse-nq/nq_query.pt \
      --spann_index renq_1 \
      --output_path ../../spann/retrieve_results/nq/renq.tsv

    cd /datacosmos/User/baoht/onesparse2/hybridsearch
    python utils/eval.py \
      --gt-path data/nq/nq_gt.tsv \
      --qrels-path spann/retrieve_results/nq/renq.tsv
}

search_nq
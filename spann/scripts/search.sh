#!/bin/bash

# Search MARCO
search_marco() {
  cd /datacosmos/User/baoht/onesparse2/hybridsearch/spann/index
  python search.py \
  --query_emb_path ../../data/msmarco/spann/marco_query.pt \
  --spann_index msmarco_1 \
  --output_path ../../spann/retrieve_results/marco/spann.tsv
  
  cd /datacosmos/User/baoht/onesparse2/hybridsearch
  python utils/eval.py \
  --gt-path data/msmarco/qrels.dev.tsv \
  --qrels-path spann/retrieve_results/marco/spann.tsv
}



# Search NQ
search_nq(){
    cd /datacosmos/User/baoht/onesparse2/hybridsearch/spann/index
    python search.py\
      --query_emb_path ../../data/nq/spann/nq_query.pt \
      --spann_index nq_1 \
      --output_path ../../spann/retrieve_results/nq/spann.tsv

    cd /datacosmos/User/baoht/onesparse2/hybridsearch
    python utils/eval.py \
      --gt-path data/nq/nq_gt.tsv \
      --qrels-path spann/retrieve_results/nq/spann.tsv
}

search_marco
# search_nq
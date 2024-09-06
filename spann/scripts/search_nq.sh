#!/bin/bash



# Search NQ
search_nq(){
    cd /datacosmos/User/baoht/onesparse2/hybridsearch/spann/index
    python search.py\
      --query_emb_path /datacosmos/User/baoht/onesparse2/Encoder/dense_tower/embeddings/nq/TEST_RUN_NQ/query/query_769.pt \
      --spann_index nq_term3_1 \
      --output_path ../../spann/retrieve_results/nq/nq_term3.tsv

    cd /datacosmos/User/baoht/onesparse2/hybridsearch
    python utils/eval.py \
      --gt-path data/nq/nq_gt.tsv \
      --qrels-path spann/retrieve_results/nq/nq_term3.tsv
}

search_nq
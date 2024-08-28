

search_nq(){
    python highbird/birds.py \
        --this_index bm25/index/nq \
        --this_index_name array_index.h5py \
        --that_index spann/index/nq_term3_invert_index \
        --that_index_name array_index.h5py \
        --spann_index spann/index/nq_term3_1/HeadIndex \
        --query_text_path data/nq/bert/nq_query_ids.json \
        --query_emb_path /datacosmos/User/baoht/onesparse2/Encoder/dense_tower/embeddings/nq/TEST_RUN_NQ/query/query_769.pt \
        --output_path highbird/retrieve_results/nq/nq_term3.tsv

    python utils/eval.py --gt-path data/nq/nq_gt.tsv --qrels-path highbird/retrieve_results/nq/nq_term3.tsv
}

search_nq
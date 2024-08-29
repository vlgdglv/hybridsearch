
search_marco(){
    python highbird/birds.py \
        --this_index bm25/index/marco \
        --this_index_name array_index.h5py \
        --that_index spann/index/marco_spann_invert_index \
        --that_index_name invert_index.pkl \
        --spann_index spann/index/msmarco_1/HeadIndex \
        --query_text_path data/msmarco/bert/dev.query.json \
        --query_emb_path data/msmarco/spann/marco_query.pt \
        --output_path highbird/retrieve_results/marco/highbird.tsv

    python utils/eval.py --gt-path data/msmarco/qrels.dev.tsv --qrels-path highbird/retrieve_results/marco/highbird.tsv
}

search_nq(){
    python highbird/birds.py \
        --this_index bm25/index/nq \
        --this_index_name array_index.h5py \
        --that_index spann/index/nq_spann_invert_index \
        --that_index_name array_index.h5py \
        --spann_index spann/index/nq_1/HeadIndex \
        --query_text_path data/nq/bert/nq_query_ids.json \
        --query_emb_path data/nq/spann/nq_query.pt \
        --output_path highbird/retrieve_results/nq/highbird_fm.tsv

    python utils/eval.py --gt-path data/nq/nq_gt.tsv --qrels-path highbird/retrieve_results/nq/highbird_fm.tsv
}

# search_marco
search_nq

search_marco(){
    python invert_index/retrieve.py \
        --index_dir  /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/index/marco/exp20240724 \
        --index_name invert_index.h5py \
        --query_text_path /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/retrieve_results/exp20240724/encode_query.json \
        --output_path invert_index/retrieve_results/marco/ii.tsv \
        --total_docs 8841823

    python utils/eval.py --gt-path data/msmarco/qrels.dev.tsv --qrels-path invert_index/retrieve_results/marco/ii.tsv
}

search_nq(){
    python invert_index/retrieve.py \
        --this_index bm25/index/nq \
        --this_index_name array_index.h5py \
        --that_index spann/index/nq_spann_invert_index \
        --that_index_name array_index.h5py \
        --spann_index spann/index/nq_1/HeadIndex \
        --query_text_path data/nq/bert/nq_query_ids.json \
        --query_emb_path data/nq/spann/nq_query.pt \
        --output_path highbird/retrieve_results/nq/highbird.tsv

    python utils/eval.py --gt-path data/nq/nq_gt.tsv --qrels-path highbird/retrieve_results/nq/highbird.tsv
}

search_marco
# search_nq
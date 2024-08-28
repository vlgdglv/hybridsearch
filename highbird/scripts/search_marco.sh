
export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

search_marco(){
    python highbird/birds.py \
        --this_index /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/index/marco/exp20240724_rebuild \
        --this_index_name invert_index.h5py \
        --that_index spann/index/marco_spann_invert_index \
        --that_index_name invert_index.pkl \
        --spann_index spann/index/msmarco_1/HeadIndex \
        --query_text_path /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/retrieve_results/exp20240724/dev.query.json \
        --query_emb_path data/msmarco/spann/marco_query.pt \
        --output_path highbird/retrieve_results/marco/os_splade.tsv

    python utils/eval.py --gt-path data/msmarco/qrels.dev.tsv --qrels-path highbird/retrieve_results/marco/os_splade.tsv
}

search_marco
# search_nq
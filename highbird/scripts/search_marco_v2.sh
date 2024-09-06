export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

OUTPUT=highbird/retrieve_results/marco/os_v2.tsv
python highbird/birds.py \
        --this_index data/msmarco/exp0724/exp20240724_rebuild  \
        --this_index_name invert_index.h5py \
        --that_index spann/index/msmarco64_spann_invert_index \
        --that_index_name invert_index.pkl \
        --spann_index spann/index/msmarco64_1/HeadIndex \
        --query_text_path data/msmarco/exp0724/dev.query.json \
        --query_emb_path data/msmarco/spann/marco_query.pt \
        --gt_path data/msmarco/qrels.dev.tsv \
        --output_path $OUTPUT  \
        --use_v2 \
        --this_weight $1 --that_weight $2 \
        --doc_emb_path  data/msmarco/spann/marco_doc_with_id769.pt \
        

# /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/index/marco/exp20240724_rebuild        
# python utils/eval.py --gt-path data/msmarco/qrels.dev.tsv --qrels-path $OUTPUT


# search_marco
# search_nq
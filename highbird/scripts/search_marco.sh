export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH

OUTPUT=highbird/retrieve_results/marco/os_bm25.tsv
python highbird/birds.py \
        --this_index bm25/index/marco \
        --this_index_name array_index.h5py \
        --that_index spann/index/msmarco_spann_invert_index \
        --that_index_name invert_index.pkl \
        --spann_index spann/index/msmarco_1/HeadIndex \
        --query_text_path data/msmarco/bert/dev.query.json \
        --query_emb_path data/msmarco/spann/marco_query.pt \
        --doc_emb_path  data/msmarco/spann/marco_doc_with_id769.pt \
        --output_path $OUTPUT  \
        --use_cmp \
        --this_weight $1 --that_weight $2 \

        # --doc_emb_path /datacosmos/User/baoht/onesparse2/Encoder/dense_tower/embeddings/msmarco/corpus/corpus_769.pt \
        # /datacosmos/User/baoht/onesparse2/Encoder/sparse_tower/index/marco/exp20240724_rebuild        

python utils/eval.py --gt-path data/msmarco/qrels.dev.tsv --qrels-path $OUTPUT

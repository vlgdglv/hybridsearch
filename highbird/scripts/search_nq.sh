export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH


python highbird/birds.py \
    --this_index bm25/index/nq \
    --this_index_name array_index.h5py \
    --that_index spann/index/nq_spann_invert_index \
    --that_index_name array_index.h5py \
    --spann_index spann/index/nq_1/HeadIndex \
    --query_text_path data/nq/bert/nq_query_ids.json \
    --query_emb_path data/nq/spann/nq_query.pt \
    --doc_emb_path data/nq/spann/nq_doc.pt \
    --output_path highbird/retrieve_results/nq/nq_os.tsv \
    --gt_path data/nq/nq_gt.tsv \
    --use_cmp \
    --this_weight $1 --that_weight $2 \

# python utils/eval.py --gt-path data/nq/nq_gt.tsv --qrels-path highbird/retrieve_results/nq/nq_os.tsv

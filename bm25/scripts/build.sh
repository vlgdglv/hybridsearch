
python bm25/main.py \
    --build_index \
    --force_rebuild \
    --tokenizer_name bert-base-uncased \
    --corpus_path data/msmarco/bert/corpus_ids.json \
    --index_path bm25/index/marco \

# python bm25/main.py \
#     --build_index \
#     --force_rebuild \
#     --tokenizer_name bert-base-uncased \
#     --corpus_path data/nq/bert/nq_doc_ids.json \
#     --index_path bm25/index/nq

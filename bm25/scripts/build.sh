
# python bm25/main.py \
#     --build_index \
#     --force_rebuild \
#     --tokenizer_name bert-base-uncased \
#     --corpus_path data/msmarco/bert/corpus_ids.json \
#     --index_path bm25/index/marco \

python bm25/main.py \
    --build_index \
    --force_rebuild \
    --do_tokenize \
    --tokenizer_name utils/bert-base-uncased-modified \
    --corpus_path data/nq/nq_doc.tsv \
    --index_path bm25/index/nq \
    --index_name array_index.h5py

    # --corpus_path data/nq/bert/nq_doc_ids.json \
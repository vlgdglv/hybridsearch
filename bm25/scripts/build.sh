python /bm25/main.py \
    --build_index \
    --tokenizer_name bert-base-uncased \
    --corpus_path data/nq_doc.tsv \
    --index_path bm25/index
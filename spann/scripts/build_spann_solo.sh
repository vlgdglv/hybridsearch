
build_nq() {
    NAME=nq 
    cd spann/index
    python build_spann_solo.py --name $NAME --corpus_path data/nq/spann/nq_doc.pt
    cd ../..
}

build_marco() {
    NAME=msmarco
    cd spann/index
    python build_spann_solo.py --name $NAME --corpus_path data/msmarco/spann/marco_doc_with_id769.pt
    cd ../..

}

# build_nq
build_marco
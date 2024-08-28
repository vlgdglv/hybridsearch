
NAME=nq_term3

cd spann/index

# python build_spann.py --name $NAME --corpus_path /datacosmos/User/baoht/onesparse2/Encoder/dense_tower/embeddings/nq/TEST_RUN_NQ/corpus/corpus_769.pt

# ./ParsePostings -i ${NAME}_1 -v float -f SPTAGFullList.bin -h SPTAGHeadVectorIDs.bin -o output.txt

cd ../..

python spann/build_invert_index.py \
    --cluster_file spann/index/${NAME}_1/output.txt \
    --index_dir spann/index/${NAME}_invert_index \
    --index_name array_index.h5py \
    --save_method h5py \
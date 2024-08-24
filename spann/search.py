import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from timeit import default_timer as timer
logger = logging.getLogger(__name__)

from bm25.BM25 import BM25Retriever
from invert_index import InvertIndex

# export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH
import SPTAG

def sptag_search(index, embedding):
    """
    return a tuple(cluster_id, score), latency
    """
    start = timer()
    # top 256 clusters
    result = index.Search(embedding, 256) # the search results are not docid, but cluster id
    latency = timer() - start
    return [(result[0][i], 1.0/(1.0+float(result[1][i]))) for i in range(len(result[1]))], latency


def prepare_query(query_text_path, query_emb_path):
    with open(query_text_path, "r") as f:
        query_texts = [json.loads(line.strip()) for line in f]
    with open(query_emb_path, "rb") as f:
        query_embeddings, qid = pickle.load(f)
    return query_texts, query_embeddings, qid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--this_index", type=str, required=False)
    parser.add_argument("--this_index_name", type=str, required=False)
    parser.add_argument("--that_index", type=str, required=False)
    parser.add_argument("--that_index_name", type=str, required=False)
    parser.add_argument("--spann_index", type=str, required=False)
    
    
    parser.add_argument("--query_text_path", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)

    parser.add_argument("--output_path", type=str, required=False)

    args = parser.parse_args()

    sptag_index = SPTAG.AnnIndex.Load(args.spann_index)
    
    with open(args.query_emb_path, "rb") as f:
        query_embeddings, qid = pickle.load(f)
    print(query_embeddings.shape)
    res, times = sptag_search(sptag_index, np.array(query_embeddings, dtype=np.float32))
    print(res[:10])
    res, times = sptag_search(sptag_index, np.array(query_embeddings[0], dtype=np.float32))
    print(res[:10])
    res, times = sptag_search(sptag_index, np.array(query_embeddings[1], dtype=np.float32))
    print(res[:10])

    # total_query = len(query_texts)
    # with open(args.output_path, "w") as f:
    #     for query_text, query_embedding in tqdm(zip(query_texts, query_embeddings), total=total_query, desc="Retrieving"):
    #         indice, scores = doer.search(query_text["text"], query_embedding)
    #         for i, (idx, score) in enumerate(zip(indice, scores)):
    #             f.write("{}\t{}\t{}\t{}\n".format(query_text["text_id"], idx, i+1, score))
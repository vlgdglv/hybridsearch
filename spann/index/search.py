import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging
import pickle
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
logger = logging.getLogger(__name__)


# export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH
import SPTAG

def sptag_search(index, embedding):
    """
    return a tuple(cluster_id, score), latency
    """
    start = timer()
    # top 256 clusters
    result = index.Search(embedding, 100) # the search results are not docid, but cluster id
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

    parser.add_argument("--spann_index", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--total_docs", type=int, required=False, default=None)
    args = parser.parse_args()

    sptag_index = SPTAG.AnnIndex.Load(args.spann_index)
    
    with open(args.query_emb_path, "rb") as f:
        query_embeddings, qid = pickle.load(f)
    
    total_query = len(query_embeddings)
    with open(args.output_path, "w") as f:
        for query_embedding, q in tqdm(zip(query_embeddings, qid), total=total_query, desc="Retrieving"):
            cluster_list, latency = sptag_search(sptag_index, query_embedding)
            cid_list, cid_score = [cluster[0] for cluster in cluster_list], [cluster[1] for cluster in cluster_list]
            for i, (idx, score) in enumerate(zip(cid_list, cid_score)):
                f.write("{}\t{}\t{}\t{}\n".format(q, idx if args.total_docs is None else idx%args.total_docs, i+1, score))
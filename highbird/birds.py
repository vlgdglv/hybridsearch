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

class Searcher:
    def __init__(self, this_index, that_index, sptag_index):
        self.this_index = this_index
        self.that_index = that_index
        self.sptag_index = sptag_index

    
    def search(self, query_text, query_embedding, topk=100):
        cluster_list, time1 = sptag_search(self.sptag_index, np.array(query_embedding.tolist(), dtype=np.float32))
        start = timer()
        cid_list, cid_score = [ cluster[0] for cluster in cluster_list], [ cluster[1] for cluster in cluster_list]
        this_postings, this_values = self.this_index.get_postings(query_text)
        cand1 = {}
        for posting, values in zip(this_postings, this_values):
            for docid, score in zip(posting, values):
                if docid in cand1:
                    cand1[docid] += score
                else:
                    cand1[docid] = score
        # cand1 = dict(sorted(cand1.items()))
        that_postings, that_values = self.that_index.get_postings(cid_list)
        cand2 = {}
        for posting, score in zip(that_postings, cid_score):
            for docid in posting:
                    cand2[docid] = score
        # cand1, cand2 = dict(sorted(cand1.items())), dict(sorted(cand2.items()))
        cand = {k: cand1.get(k, 0) + cand2.get(k, 0) * 10000 for k in set(cand1.keys()) & set(cand2.keys())}
        cand = dict(sorted(cand.items(), key=lambda x: x[1], reverse=True))
        docs, scores = np.array(list(cand.keys())[:topk]), np.array(list(cand.values())[:topk])
        end = timer()
        time2 = end - start
        return docs, scores

    @classmethod
    def build(cls, this_index_path, this_index_name, that_index_path, that_index_name, spann_index_path):
        sptag_index = SPTAG.AnnIndex.Load(spann_index_path)
        this_index = BM25Retriever(this_index_path, this_index_name)
        that_index = InvertIndex(that_index_path, that_index_name)
        return cls(this_index, that_index, sptag_index)

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

    doer = Searcher.build(args.this_index, args.this_index_name,
                          args.that_index, args.that_index_name, args.spann_index)
    
    query_texts, query_embeddings, qid = prepare_query(args.query_text_path, args.query_emb_path)
    total_query = len(query_texts)
    with open(args.output_path, "w") as f:
        for query_text, query_embedding in tqdm(zip(query_texts, query_embeddings), total=total_query, desc="Retrieving"):
            indice, scores = doer.search(query_text["text"], query_embedding)
            for i, (idx, score) in enumerate(zip(indice, scores)):
                f.write("{}\t{}\t{}\t{}\n".format(query_text["text_id"], idx, i+1, score))
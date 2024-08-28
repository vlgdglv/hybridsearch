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
from collections import defaultdict

logger = logging.getLogger(__name__)

from bm25.BM25 import BM25Retriever
from invert_index import InvertIndex


if __name__  == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--index_dir", type=str, required=False)
    parser.add_argument("--index_name", type=str, required=False)
    
    parser.add_argument("--total_docs", type=int, required=False)
    parser.add_argument("--query_text_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)

    args = parser.parse_args()

    ii = InvertIndex(args.index_dir, args.index_name, index_dim=30522)
    ii.engage_numba()

    
    query_list = []
    with open(args.query_text_path, "r") as f:
        for line in f:
            query_list.append(json.loads(line)) 


    with open(args.output_path, "w") as f:
        for query in tqdm(query_list, total=len(query_list), desc="Retrieving"):
            scores = ii.match_numba(
                ii.numba_index_ids,
                ii.numba_index_values,
                query["text"], 
                args.total_docs,
                query["value"]
            )

            indice, scores = ii.select_topk(scores)
            for i, (idx, score) in enumerate(zip(indice, scores)):
                f.write("{}\t{}\t{}\t{}\n".format(query["text_id"][0], idx, i+1, score))
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import logging

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

from BM25 import BM25Retriever
from tokenizer import Tokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, required=False)
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--corpus_path", type=str, required=False)
    parser.add_argument("--index_path", type=str, required=False)
    parser.add_argument("--index_name", type=str, required=False)
    parser.add_argument("--do_tokenize", action="store_true")
    parser.add_argument("--index_file_name", type=str, default="array_index", required=False)
    parser.add_argument("--force_rebuild", action="store_true")
    
    # BM25 params
    parser.add_argument("--method", type=str, default="lucene", required=False)
    parser.add_argument("--k1", type=float, default=1.2, required=False)
    parser.add_argument("--b", type=float, default=0.75, required=False)
    parser.add_argument("--delta", type=float, default=0.5, required=False)
    
    parser.add_argument("--do_retrieve", action="store_true")
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)

    args = parser.parse_args()

    print(args)
    
    if args.build_index:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        bm25 = BM25Retriever(args.index_path, args.index_name, method=args.method, k1=args.k1, b=args.b, delta=args.delta, 
                             force_rebuild=args.force_rebuild)

        corpus_idx, corpus_list = [], []
        if args.do_tokenize:
            with open(args.corpus_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Tokenizing"):
                    cid, text = line.strip().split('\t')
                    corpus_idx.append(cid)
                    corpus_list.append(tokenizer.encode(text))
        else:
            with open(args.corpus_path, "r") as f:
                for line in tqdm(f, desc="Loading"):
                    content = json.loads(line.strip())
                    corpus_idx.append(int(content["text_id"]))
                    corpus_list.append(content["text"])
        print("Corpus Loaded: {}".format(len(corpus_idx)))
        # print("Text encode done")
        bm25.index(corpus_idx, corpus_list, tokenizer.vocab.values())

    if args.do_retrieve:
        # stemmer = Stemmer.Stemmer("english")
        # token = Tokenizer(tokenizer_path=args.tokenizer_path, stemmer=stemmer)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        bm25 = BM25Retriever(args.index_path, args.index_name, k1=args.k1, b=args.b, delta=args.delta)

        query_idx, query_list = [], []
        if args.do_tokenize:
            with open(args.query_path, "r", encoding="utf-8") as f:
                for line in f:
                    qid, text = line.strip().split('\t')
                    query_idx.append(qid)
                    query_list.append(tokenizer.encode(text))
        else:
            with open(args.query_path, "r") as f:
                for line in f:
                    content = json.loads(line.strip())
                    query_idx.append(int(content["text_id"]))
                    query_list.append(content["text"])
        print("Query Loaded: {}".format(len(query_idx)))
        
        bm25.invert_index.engage_numba()
        with open(args.output_path, "w", encoding="utf-8") as out:
            for qid, query in tqdm(zip(query_idx, query_list), desc="Retrieving", total=len(query_idx)):
                indices, scores = bm25.retrieve(np.array(query))
                for rank, (docid, score) in enumerate(zip(indices, scores)):
                    out.write(f"{qid}\t{docid}\t{rank+1}\t{score}\n")
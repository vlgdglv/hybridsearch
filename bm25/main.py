import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import Stemmer
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from BM25 import BM25Retriever
from tokenizer import Tokenizer


def train_tokenizer(train_data_files, save_dir):

    stemmer = Stemmer.Stemmer("english")
    token = Tokenizer(stemmer=stemmer)
    logger.info("Training tokenizer with files: {}".format(train_data_files))

    train_texts = []
    for files in train_data_files:
        text_list = []
        with open(files, "r", encoding="utf-8") as f:
            for line in f:
                text_list.append(line.strip().split('\t')[1])
        train_texts.extend(text_list)
    logger.info("Total train texts: {}".format(len(train_texts)))    
    
    token.train(train_texts, save_dir)
    logger.info("Tokenizer trained, saved to {}".format(save_dir))

parser = argparse.ArgumentParser()
parser.add_argument("--train_tokenizer", action="store_true")
parser.add_argument("--train_tok_files", type=str, nargs="+", required=False)
parser.add_argument("--tok_save_dir", type=str, required=False)

parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--build_index", action="store_true")
parser.add_argument("--corpus_path", type=str, required=False)
parser.add_argument("--index_path", type=str, required=False)

parser.add_argument("--do_retrieve", action="store_true")
parser.add_argument("--query_path", type=str, required=False)
parser.add_argument("--output_path", type=str, required=False)

args = parser.parse_args()

if __name__ == "__main__":
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if args.train_tokenizer:
        train_tokenizer(args.train_data_files, args.save_dir)    
    
    if args.build_index:
        stemmer = Stemmer.Stemmer("english")
        token = Tokenizer(tokenizer_path=args.tokenizer_path, stemmer=stemmer)
        bm25 = BM25Retriever(args.index_path)

        corpus_idx, corpus_list = [], []
        with open(args.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                cid, text = line.strip().split('\t')
                corpus_idx.append(cid)
                corpus_list.append(text)

        text_ids = token.encode(corpus_list)
    
        print("Text encode done")
        bm25.index(corpus_idx, text_ids, token.get_vocab_ids())

    if args.do_retrieve:
        stemmer = Stemmer.Stemmer("english")
        token = Tokenizer(tokenizer_path=args.tokenizer_path, stemmer=stemmer)
        bm25 = BM25Retriever(args.index_path)

        query_idx, query_list = [], []
        with open(args.query_path, "r", encoding="utf-8") as f:
            for line in f:
                qid, text = line.strip().split('\t')
                query_idx.append(qid)
                query_list.append(text)
        print("Query Loaded: {}".format(len(query_idx)))
        
        # bm25.invert_index.engage_numba()
        query_ids = token.encode(query_list)
        with open(args.output_path, "w", encoding="utf-8") as out:
            for qid, query in tqdm(zip(query_idx, query_ids), desc="Retrieving", total=len(query_idx)):
                indices, scores = bm25.retrieve(np.array(query))
                for rank, (docid, score) in enumerate(zip(indices, scores)):
                    out.write(f"{qid}\t{docid}\t{rank+1}\t{score}\n")
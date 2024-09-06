import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
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

# export PYTHONPATH=/home/aiscuser/SPTAG/Release:PYTHONPATH
import SPTAG
from numba import njit
from numba.typed import Dict, List
from numba import types

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
    def __init__(self, this_index, that_index, sptag_index, doc_embedding=None,
                 this_weight=1.0,
                 that_weight=10000.0,
                 use_cmp=False):
        self.this_index = this_index
        self.that_index = that_index
        self.sptag_index = sptag_index
        self.corpus_length = 1 # self.this_index.get_total_docs()
        self.this_index.engage_numba()

        self.this_weight = this_weight
        self.that_weight = that_weight

        self.doc_embedding = doc_embedding
        self.use_cmp = use_cmp
    
    def search(self, query, query_embedding, topk=100):
        query_text = query["text"]
        query_value = query["value"] if "value" in query else [1.0 for _ in range(len(query_text))]
        # print(query_value)
        ######## Simple but slow merge ######## 
        def simple_merge(this_postings, this_values, that_postings, that_values):
            cand1 = {}
            for posting, values in zip(this_postings, this_values):
                for docid, score in zip(posting, values):
                    try:
                        cand1[docid] += score
                    except:
                        cand1[docid] = score
            # t11 = timer() 
            # cand1 = dict(sorted(cand1.items()))
            cand2 = {}
            # t21 = timer()
            for posting, score in zip(that_postings, that_values):
                for docid in posting:
                    cand2[docid] = score
            # cand1, cand2 = dict(sorted(cand1.items())), dict(sorted(cand2.items()))
            # t2 = timer()
            cand = {k: cand1.get(k, 0) + cand2.get(k, 0) * 10000 for k in set(cand1.keys()) & set(cand2.keys())}
            cand = dict(sorted(cand.items(), key=lambda x: x[1], reverse=True))
            return np.array(list(cand.keys())[:topk]), np.array(list(cand.values())[:topk])

        # docs, scores = simple_merge(this_postings, this_values, that_postings, that_values)
        ######## Some fast merge ########
        def some_fast_merge(this_postings, this_values, that_postings, that_values):
            current_docid = -1
            this_cand, that_cand = defaultdict(float), defaultdict(float)

            this_cursor, that_cursor = [0 for _ in range(len(this_postings))], [0 for _ in range(len(that_postings))]
            while True:
                this_min_id, this_min_posting_idx, this_min_value = 10000000000, [], 0
                for i, posting_id in enumerate(this_postings):
                    cursor, len_cur_posting = this_cursor[i], len(posting_id) 
                    while cursor < len_cur_posting and posting_id[cursor] < current_docid:
                        cursor += 1
                    this_cursor[i] = cursor
                    if cursor < len_cur_posting:
                        if posting_id[cursor] < this_min_id:
                            this_min_id = posting_id[cursor]
                            this_min_posting_idx = [i]
                            this_min_value = this_values[i][cursor]
                        elif posting_id[cursor] == this_min_id:
                            this_min_value += this_values[i][cursor]
                            this_min_posting_idx.append(i)

                that_min_id, that_min_posting_idx, that_min_value = 10000000000, [], 0
                for i, posting_id in enumerate(that_postings):
                    cursor, len_cur_posting = that_cursor[i], len(posting_id)
                    while cursor < len_cur_posting and posting_id[cursor] < current_docid:
                        cursor += 1
                    that_cursor[i] = cursor
                    if cursor < len_cur_posting: 
                        if  posting_id[cursor] < that_min_id:
                            that_min_id = posting_id[cursor]
                            that_min_posting_idx = [i]
                            that_min_value = that_values[i]  # use value for single posting list
                        elif posting_id[cursor] == that_min_id:
                            that_min_value += that_values[i] # use value for single posting list
                            that_min_posting_idx.append(i)
                
                if this_min_id == 10000000000 and that_min_id == 10000000000:
                    break

                if this_min_id == that_min_id:
                    this_cand[this_min_id] += this_min_value
                    that_cand[that_min_id] += that_min_value
                    current_docid = this_min_id
                    for i in this_min_posting_idx:
                        this_cursor[i] += 1
                    for i in that_min_posting_idx:
                        that_cursor[i] += 1
                else:
                    current_docid = max(this_min_id, that_min_id)

            final_dict = {}
            for k in this_cand.keys():
                final_dict[k] = this_cand[k] + that_cand[k] * 10000

            final_dict = dict(sorted(final_dict.items(), key=lambda x: x[1], reverse=True))
        
            return np.array(list(final_dict.keys())[:topk]), np.array(list(final_dict.values())[:topk])
        
        # Begin Here
        cluster_list, time1 = sptag_search(self.sptag_index, np.array(query_embedding.tolist(), dtype=np.float32))
        cid_list, cid_score = [cluster[0] for cluster in cluster_list], [ cluster[1] for cluster in cluster_list]

        start = timer()
        this_postings, this_values = self.this_index.get_postings(query_text)
        t1 = timer()
        cand1 = {}
        cand1 = Dict.empty(
            key_type=types.int64,
            value_type=types.float64,
        )
        add_scores(cand1, this_postings, this_values, query_value)
        t11 = timer() 
        cand2 = defaultdict(float)
        that_postings, that_values = self.that_index.get_postings(cid_list)
        t21 = timer()
        

        cand2_key_list = []
        for posting, score in zip(that_postings, cid_score):
            for docid in posting:
                    cand2_key_list.append(docid)
                    cand2[docid] = score 
        if self.use_cmp:
            pass
        else:    
            cand_doc_embd = self.doc_embedding[cand2_key_list]
            cand2_dis = 1 / np.sum((query_embedding - cand_doc_embd) ** 2, axis=1)
            cand2 = dict(zip(cand2_key_list, cand2_dis))

        t2 = timer()
    
        cand = {k:  cand1.get(k, 0) * self.this_weight
                  + cand2.get(k, 0) * self.that_weight
                 for k in set(cand1.keys()) & set(cand2.keys())} # & 

        cand = dict(sorted(cand.items(), key=lambda x: x[1], reverse=True))
        
        docs, scores = np.array(list(cand.keys())[:topk]), np.array(list(cand.values())[:topk])
        # docs, scores = simple_merge( this_postings, this_values, that_postings, cid_score )
        end = timer()
        
        time_dict = {"total": end - start, 
                     "sptag": time1, 
                     "retrieve_postings1": t1 - start, 
                     "merge_postings1": t11 - t1,
                     "retrieve_postings2": t21 - t11,
                     "merge_postings2": t2 - t21,
                     "get_topk": end - t2}
        
        return docs, scores, time_dict #, score_dict


    @classmethod
    def build(cls, this_index_path, this_index_name, 
              that_index_path, that_index_name, spann_index_path, doc_emb_path,
              this_weight, that_weight, use_cmp, use_v2):
        sptag_index = SPTAG.AnnIndex.Load(spann_index_path)
        if not use_v2:
            this_index = BM25Retriever(this_index_path, this_index_name)
        else:
            this_index = InvertIndex(this_index_path, this_index_name, index_dim=30522) # temp magic number

        that_index = InvertIndex(that_index_path, that_index_name)
        if doc_emb_path is not None:
            with open(doc_emb_path, "rb") as f:
                doc_embedding,_ = pickle.load(f)
        else:
            doc_embedding = None
        return cls(this_index, that_index, sptag_index, doc_embedding,this_weight, that_weight, use_cmp)


def eval_recall(gt_path, result_lists):
    recall = 0.0
    with open(gt_path, "r", encoding="utf8") as f_gt: 
        gt_tsvreader = csv.reader(f_gt, delimiter="\t")
        # result_tsvreader = csv.reader(f_result, delimiter="\t")
        
        gt_dict = {}
        for [gt_qid, gt_docid] in gt_tsvreader:
            if int(gt_qid) not in gt_dict.keys():
                gt_dict[int(gt_qid)] = [int(gt_docid)]
            else:
                gt_dict[int(gt_qid)].append(int(gt_docid))
        
        result_dict = {}
        for [result_qid, result_docid, _, _] in result_lists:
            if int(result_qid) not in result_dict.keys():
                result_dict[int(result_qid)] = [int(result_docid)]
            else:
                result_dict[int(result_qid)].append(int(result_docid))

        idx = 0
        for qid in gt_dict.keys():
            gt_list = gt_dict[qid]
            idx += 1
            try:
                result_list = result_dict[qid]
            except:
                result_list = []

            # calculate recall
            recall += len(set(gt_list).intersection(set(result_list)))/len(set(gt_list))

    return recall/idx
    

@njit
def add_scores(cand, postings, scores, values):
    for i in range(len(postings)):
        for j in range(len(postings[i])):
            docid = postings[i][j]
            score = scores[i][j] * values[i]
            if docid in cand:
                cand[docid] += score
            else:
                cand[docid] = score
    

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
    
    parser.add_argument("--this_weight", type=float, default=1, required=False)
    parser.add_argument("--that_weight", type=float, default=10000, required=False)

    parser.add_argument("--query_text_path", type=str, required=False)
    parser.add_argument("--query_emb_path", type=str, required=False)
    parser.add_argument("--doc_emb_path", type=str, required=False)

    parser.add_argument("--use_cmp", action="store_true", required=False)
    parser.add_argument("--use_v2", action="store_true", required=False)

    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--gt_path", type=str, required=False)
    
    args = parser.parse_args()


    doer = Searcher.build(args.this_index, args.this_index_name,
                          args.that_index, args.that_index_name, 
                          args.spann_index, args.doc_emb_path,
                          args.this_weight, args.that_weight,
                          args.use_cmp, args.use_v2)
    
    query_texts, query_embeddings, qid = prepare_query(args.query_text_path, args.query_emb_path)
    total_query = len(query_texts)
    
    total_time_dict = {
        "total": 0,
        "sptag": 0, 
        "retrieve_postings1": 0, 
        "merge_postings1": 0,
        "retrieve_postings2": 0, 
        "merge_postings2": 0,
        "get_topk": 0,
    }


    res_list = []
    with open(args.output_path, "w") as f:
        for query_text, query_embedding in tqdm(zip(query_texts, query_embeddings), total=total_query, desc="Retrieving"):
            indice, scores, time_dict = doer.search(query_text, query_embedding)
            for i, (idx, score) in enumerate(zip(indice, scores)):
                f.write("{}\t{}\t{}\t{}\n".format(query_text["text_id"], idx, i+1, score))
                res_list.append([query_text["text_id"], idx, i+1, score])
            for key, value in time_dict.items():
                total_time_dict[key] += value

    for key, value in total_time_dict.items():
        print("{}:\t {} ms".format(key, 1000 * value / total_query))      
    
    rec100 = eval_recall(args.gt_path, res_list)
    print("Recall@100: ", rec100)
import os
import h5py
import json
import array
import numba
import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from numba.typed import Dict, List
from numba import types

# from posting_list import PostingList

logger = logging.getLogger(__name__)

def create_int_array():
    return array.array('i')

def create_float_array():
    return array.array('f')

class InvertIndex:
    def __init__(self, 
                 index_path=None, 
                 file_name="array_index",
                 force_rebuild=False,
                 save_method="pickle",
                 index_dim=None):
        os.makedirs(index_path, exist_ok=True)
        self.save_method = save_method
        
        self.file_path = os.path.join(index_path, file_name)
        self.index_path = index_path
        suffix = file_name.split(".")[-1]
        if suffix == "h5py":
            if os.path.exists(self.file_path) and not force_rebuild:
                print("Loading index from {}".format(self.file_path))
                with h5py.File(self.file_path, "r") as f:
                    self.index_ids = dict()
                    self.index_values = dict()
                    dim = f["dim"][()] if index_dim is None else index_dim
                    for key in tqdm(range(dim), desc="Loading index"):
                        try:
                            self.index_ids[key] = np.array(f["index_ids_{}".format(key)], dtype=np.int32)
                            self.index_values[key] = np.array(f["index_values_{}".format(key)], dtype=np.float32)
                        except:
                            self.index_ids[key] = np.array([], dtype=np.int32)
                            self.index_values[key] = np.array([], dtype=np.float32)
                    try:
                        self.total_docs = f["total_docs"][()]
                    except:
                        self.total_docs = -1
                    f.close()
                print("Index loaded")
            else:
                print("Building index")
                self.index_ids = defaultdict(lambda: array.array('i'))
                self.index_values = defaultdict(lambda: array.array('f'))
                self.total_docs = 0
        elif suffix == "pkl":
            if os.path.exists(self.file_path) and not force_rebuild:
                print("Loading index from {}".format(self.file_path))
                with open(self.file_path, "rb") as f:
                    index_ids, index_values, total_docs = pickle.load(f)
                    self.index_ids, self.index_values, self.total_docs = index_ids, index_values, total_docs
                print("Index loaded, total docs: {}".format(self.total_docs))
            else:
                print("Building index")
                self.index_ids = defaultdict(create_int_array)
                self.index_values = defaultdict(create_float_array)
                self.total_docs = 0
        
        self.numba = False

    def add_batch_item(self, col, row, value):
        for r, c, v in zip(row, col, value):
            self.index_ids[c].append(int(r))
            self.index_values[c].append(v)
        self.total_docs += len(set(row))

    def add_item(self, col, row, value):
        self.index_ids[col].append(int(row))
        self.index_values[col].append(value)
        self.total_docs += 1

    def save(self):
        print("Converting to numpy")
        for key in tqdm(list(self.index_ids.keys()), desc="Converting to numpy"):
            self.index_ids[key] = np.array(self.index_ids[key], dtype=np.int32)
            self.index_values[key] = np.array(self.index_values[key], dtype=np.float32)
            
        if self.save_method == "h5py":
            print("Save index to {}".format(self.file_path))
            with h5py.File(self.file_path, "w") as f:
                f.create_dataset("dim", data=len(self.index_ids.keys()))
                f.create_dataset("total_docs", data=self.total_docs)    
                for key in tqdm(self.index_ids.keys(), desc="Saving"):
                    f.create_dataset("index_ids_{}".format(key), data=self.index_ids[key])
                    f.create_dataset("index_values_{}".format(key), data=self.index_values[key])
                f.close()
        elif self.save_method == "pickle":
            print("Save index to {}".format(self.file_path))
            with open(self.file_path, "wb") as f:
                pickle.dump((self.index_ids, self.index_values, self.total_docs), f)
        print("Index saved")
        index_dist = {}
        for k, v in self.index_ids.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))

    def save_json(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

    def __len__(self):
        return len(self.index_ids)
    
    def engage_numba(self):
        self.numba_index_ids = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
        self.numba_index_values = Dict.empty(
            key_type=types.int64,
            value_type=types.float64[:]
        )
        self.numba = True

        for k, v in self.index_ids.items():
            self.numba_index_ids[k] = np.array(v, dtype=np.int64)
        for k, v in self.index_values.items():
            self.numba_index_values[k] = np.array(v, dtype=np.float64)
        print("Numba engaged")

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def match_numba(numba_index_ids: numba.typed.Dict,
              numba_index_values: numba.typed.Dict, 
              query_ids: np.ndarray, 
              corpus_size: int, 
              query_values: np.ndarray = None):
        scores = np.zeros(corpus_size, dtype=np.float32)
        N = len(query_ids)
        for i in range(N):
            query_idx, query_value = query_ids[i], query_values[i] if query_values is not None else 1.0
            try:
                retrieved_indices = numba_index_ids[query_idx]
                retrieved_values = numba_index_values[query_idx]
            except:
                continue
            for j in numba.prange(len(retrieved_indices)):
                scores[retrieved_indices[j]] += query_value * retrieved_values[j]
        return scores

    def match(self, query_ids, corpus_size, query_values=None):
        scores = np.zeros(corpus_size, dtype=np.float32)
        N = len(query_ids)
        for i in range(N):
            query_idx, query_value = query_ids[i], query_values[i] if query_values is not None else 1.0
            if query_idx not in self.index_ids:
                continue
            retrieved_indices = self.index_ids[query_idx]
            retrieved_values = self.index_values[query_idx]
            for j in numba.prange(len(retrieved_indices)):
                scores[retrieved_indices[j]] += query_value * retrieved_values[j]
        return scores

    def select_topk(self, scores, threshold=0.0, topk=100):
        filtered_indices = np.argwhere(scores > threshold)[:, 0]
        scores = scores[filtered_indices]
        if len(scores) > topk:
            top_indices = np.argpartition(scores, -topk)[-topk:]
            filtered_indices, scores = filtered_indices[top_indices], scores[top_indices]
        sorted_indices = np.argsort(-scores)
        return filtered_indices[sorted_indices], scores[sorted_indices]

    def get_postings(self, query):
        posting_list, posting_value = [], [] 
        for i in range(len(query)):
            query_idx = query[i]
            if query_idx in self.index_ids.keys():
                posting_list.append(self.index_ids[query_idx])
                posting_value.append(self.index_values[query_idx])
        return posting_list, posting_value


    def numba_get_postings(self, query):
        posting_list, posting_value = List.empty_list(types.int32[:]), List.empty_list(types.float64[:])
        for i in range(len(query)):
            query_idx = query[i]
            if query_idx not in self.numba_index_ids.keys():
                continue
            posting_list.append(self.numba_index_ids[query_idx])
            posting_value.append(self.numba_index_values[query_idx])
        return posting_list, posting_value


import os
import h5py
import json
import array
import numba
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

class InvertIndex:
    def __init__(self, 
                 index_path=None, 
                 file_name="array_index",
                 force_rebuild=False):
        os.makedirs(index_path, exist_ok=True)
        self.file_path = os.path.join(index_path, "{}.h5py".format(file_name))
        self.index_path = index_path
        if os.path.exists(self.file_path) and not force_rebuild:
            logger.info("Loading index from {}".format(self.file_path))
            with h5py.File(self.file_path, "r") as f:
                self.index_ids = dict()
                self.index_values = dict()
                dim = f["dim"][()]
                for key in tqdm(range(dim), desc="Loading index"):
                    try:
                        self.index_ids[key] = np.array(f["index_ids_{}".format(key)], dtype=np.int32)
                        self.index_values[key] = np.array(f["index_values_{}".format(key)], dtype=np.float32)
                    except:
                        self.index_ids[key] = np.array([], dtype=np.int32)
                        self.index_values[key] = np.array([], dtype=np.float32)
                f.close()
            logger.info("Index loaded")
            # self.total_docs = f["total_docs"][()]
            self.total_docs = 152027 # Very wrong, temp use
        else:
            logger.info("Building index")
            self.index_ids = defaultdict(lambda: array.array('i'))
            self.index_values = defaultdict(lambda: array.array('f'))
            self.total_docs = 0

    def add_item(self, row, col, value):
        for r, c, v in zip(row, col, value):
            self.index_ids[c].append(int(r))
            self.index_values[c].append(v)
        self.total_docs += 1


    def save(self):
        logger.info("Converting to numpy")
        for key in tqdm(list(self.index_ids.keys()), desc="Converting to numpy"):
            self.index_ids[key] = np.array(self.index_ids[key], dtype=np.int32)
            self.index_values[key] = np.array(self.index_values[key], dtype=np.float32)

        logger.info("Save index to {}".format(self.file_path))
        with h5py.File(self.file_path, "w") as f:
            f.create_dataset("dim", data=len(self.index_ids.keys()))
            f.create_dataset("total_docs", data=self.total_docs)    
            for key in tqdm(self.index_ids.keys(), desc="Saving"):
                f.create_dataset("index_ids_{}".format(key), data=self.index_ids[key])
                f.create_dataset("index_values_{}".format(key), data=self.index_values[key])
            f.close()

        index_dist = {}
        for k, v in self.index_ids.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))

    def __len__(self):
        return len(self.index_ids)
    
    def engage_numba(self):
        self.numba_index_ids = numba.typed.Dict()
        self.numba_index_values = numba.typed.Dict()
        self.numba = True

        for k, v in self.index_ids.items():
            self.numba_index_ids[k] = v
        for k, v in self.index_values.items():
            self.numba_index_values[k] = v
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
            if query_idx not in numba_index_ids:
                continue
            retrieved_indices = numba_index_ids[query_idx]
            retrieved_values = numba_index_values[query_idx]
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
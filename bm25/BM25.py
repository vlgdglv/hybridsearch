import logging
import numpy as np
from tqdm import tqdm
from bm25.scoring import select_idf_scorer, select_tfc_scorer

from invert_index import InvertIndex

logger = logging.getLogger(__name__)

class BM25Retriever:
    def __init__(self,
                 index_path,
                 index_name,
                 method="lucene",
                 k1=1.2,
                 b=0.75,
                 delta=0.5,
                 force_rebuild=False):
        self.method = method
        self.k1 = k1
        self.b = b
        self.delta = delta

        self.index_path = index_path
        self.invert_index = InvertIndex(index_path, index_name, force_rebuild=force_rebuild)
    

    def retrieve(self, query_ids, topk=100, threshold=0.0):
        if self.invert_index.numba:
            scores = self.invert_index.match_numba(
                self.invert_index.numba_index_ids,
                self.invert_index.numba_index_values,
                query_ids, corpus_size=self.invert_index.total_docs)
            
        else:
            scores = self.invert_index.match(query_ids, corpus_size=self.invert_index.total_docs)    
        filtered_indices = np.argwhere(scores > threshold)[:, 0]
        scores = scores[filtered_indices]
        # select top K
        if len(scores) > topk:
            top_indices = np.argpartition(scores, -topk)[-topk:]
            filtered_indices, scores = filtered_indices[top_indices], scores[top_indices]
        sorted_indices = np.argsort(-scores)

        return filtered_indices[sorted_indices], scores[sorted_indices]

    
    def index(self, 
              corpus_index,
              corpus_ids,
              vocab_ids):
        n_docs, n_vocab = len(corpus_ids), len(vocab_ids)
        logger.debug("Building with n_docs: {}, n_vocab: {}".format(n_docs, n_vocab))
        avg_doc_len = np.mean([len(doc) for doc in corpus_ids])
        
        doc_frequencies = self._calc_doc_frequencies(corpus_ids, vocab_ids)
        idf_array = self._calc_idf_array(select_idf_scorer(self.method), doc_frequencies, n_docs)

        calc_tfc_fn = select_tfc_scorer(self.method)

        for doc_idx, token_ids in tqdm(zip(corpus_index, corpus_ids), desc="BM25 Indexing", total=n_docs):
            doc_len = len(token_ids)

            unique_tokens = set(token_ids)
            tf_dict = {tid: token_ids.count(tid) for tid in unique_tokens}
            token_in_doc = np.array(list(tf_dict.keys()))
            tf_array = np.array(list(tf_dict.values()))

            tfc = calc_tfc_fn(tf_array, doc_len, avg_doc_len, self.k1, self.b, self.delta)
            idf = idf_array[token_in_doc]

            # [DOC_LEN]
            scores = tfc * idf 
            self.invert_index.add_batch_item(token_in_doc, [doc_idx for _ in range(len(token_in_doc))], scores)

        self.invert_index.save()
    
    def get_postings(self, query):
        return self.invert_index.get_postings(query)
    
    def _calc_doc_frequencies(self, corpus_ids, vocab_ids):
        vocab_set = set(vocab_ids)
    
        doc_freq_dict = {token_id: 0 for token_id in vocab_ids}

        for doc in corpus_ids:
            for doc_token in vocab_set.intersection(set(doc)):
                doc_freq_dict[doc_token] += 1
        return doc_freq_dict
    
    def _calc_idf_array(self, idf_calc_fn, doc_frequencies, n_docs):
        idf_array = np.zeros(len(doc_frequencies))
        for token_id, doc_freq in doc_frequencies.items():
            idf_array[token_id] = idf_calc_fn(doc_freq, n_docs)

        return idf_array
    
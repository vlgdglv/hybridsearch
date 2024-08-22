import math
import numpy as np



def _score_idf_robertson(df, N, allow_negative=False):
    """
    Computes the inverse document frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    inner = (N - df + 0.5) / (df + 0.5)
    if not allow_negative and inner < 1:
        inner = 1

    return math.log(inner)


def _score_idf_lucene(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(1 + (N - df + 0.5) / (df + 0.5))


def _score_idf_atire(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log(N / df)


def _score_idf_bm25l(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / (df + 0.5))


def _score_idf_bm25plus(df, N):
    """
    Computes the inverse document frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return math.log((N + 1) / df)

def _score_tfc_robertson(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Robertson+ (original) variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return tf_array / (k1 * ((1 - b) + b * l_d / l_avg) + tf_array)


def _score_tfc_lucene(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using Lucene variant (accurate)
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    return _score_tfc_robertson(tf_array, l_d, l_avg, k1, b)


def _score_tfc_atire(tf_array, l_d, l_avg, k1, b, delta=None):
    """
    Computes the term frequency component of the BM25 score using ATIRE variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    # idf component is given by the idf_array
    # we calculate the term-frequency component (tfc)
    return (tf_array * (k1 + 1)) / (tf_array + k1 * (1 - b + b * l_d / l_avg))


def _score_tfc_bm25l(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25L variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    c_array = tf_array / (1 - b + b * l_d / l_avg)
    return ((k1 + 1) * (c_array + delta)) / (k1 + c_array + delta)


def _score_tfc_bm25plus(tf_array, l_d, l_avg, k1, b, delta):
    """
    Computes the term frequency component of the BM25 score using BM25+ variant
    Implementation: https://cs.uwaterloo.ca/~jimmylin/publications/Kamphuis_etal_ECIR2020_preprint.pdf
    """
    num = (k1 + 1) * tf_array
    den = k1 * (1 - b + b * l_d / l_avg) + tf_array
    return (num / den) + delta

def select_idf_scorer(method) -> callable:
    if method == "robertson":
        return _score_idf_robertson
    elif method == "lucene":
        return _score_idf_lucene
    elif method == "atire":
        return _score_idf_atire
    elif method == "bm25l":
        return _score_idf_bm25l
    elif method == "bm25+":
        return _score_idf_bm25plus
    else:
        error_msg = f"Invalid score_idf_inner value: {method}. Choose from 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'."
        raise ValueError(error_msg)
    
def select_tfc_scorer(method) -> callable:
    if method == "robertson":
        return _score_tfc_robertson
    elif method == "lucene":
        return _score_tfc_lucene
    elif method == "atire":
        return _score_tfc_atire
    elif method == "bm25l":
        return _score_tfc_bm25l
    elif method == "bm25+":
        return _score_tfc_bm25plus
    else:
        error_msg = f"Invalid score_tfc value: {method}. Choose from 'robertson', 'lucene', 'atire'."
        raise ValueError(error_msg)

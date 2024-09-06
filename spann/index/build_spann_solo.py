import pickle
import numpy as np
import argparse
import SPTAG

 
def build_index(corpus_name, corpus_path):
    '''
    if corpus_name == "nq":
        with open(corpus_path, "rb") as f:
            corpus, _ = pickle.load(f)
        corpus = np.array(corpus)
    elif corpus_name == "msmarco":
        corpus = []
        for i in range(10):
            with open(corpus_path + "%d.pt" % i, "rb") as f:
                embedding, _ = pickle.load(f)
                corpus.append(embedding)
        corpus = np.vstack(corpus)
    else:
        raise NotImplementedError
    '''

    with open(corpus_path, "rb") as f:
        corpus, id_list = pickle.load(f)
    corpus = np.array(corpus)
    
    # corpus = read_fbin(corpus_path)

    vector_number, vector_dim = corpus.shape
    print(vector_number, vector_dim)

    print("Build index Begin !!!")

    index = SPTAG.AnnIndex('SPANN', 'Float', vector_dim)

    index = SPTAG.AnnIndex('SPANN', 'Float', vector_dim)

    index.SetBuildParam("IndexAlgoType", "BKT", "Base")
    index.SetBuildParam("IndexDirectory", corpus_name + "_8", "Base")
    index.SetBuildParam("DistCalcMethod", "L2", "Base")

    index.SetBuildParam("isExecute", "true", "SelectHead")
    index.SetBuildParam("TreeNumber", "1", "SelectHead")
    index.SetBuildParam("BKTKmeansK", "32", "SelectHead")
    index.SetBuildParam("BKTLeafSize", "8", "SelectHead")
    index.SetBuildParam("SamplesNumber", "10000", "SelectHead")
    index.SetBuildParam("SelectThreshold", "50", "SelectHead") 
    index.SetBuildParam("SplitFactor", "6", "SelectHead")    
    index.SetBuildParam("SplitThreshold", "100", "SelectHead")  
    index.SetBuildParam("Ratio", "0.1", "SelectHead")   
    index.SetBuildParam("NumberOfThreads", "16", "SelectHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "SelectHead")

    index.SetBuildParam("isExecute", "true", "BuildHead")
    index.SetBuildParam("NeighborhoodSize", "32", "BuildHead")
    index.SetBuildParam("TPTNumber", "64", "BuildHead")
    index.SetBuildParam("TPTLeafSize", "2000", "BuildHead")
    index.SetBuildParam("MaxCheck", "8192", "BuildHead")
    index.SetBuildParam("MaxCheckForRefineGraph", "8192", "BuildHead")
    index.SetBuildParam("RefineIterations", "3", "BuildHead")
    index.SetBuildParam("NumberOfThreads", "16", "BuildHead")
    index.SetBuildParam("BKTLambdaFactor", "-1", "BuildHead")

    index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
    index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
    index.SetBuildParam("InternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("ReplicaCount", "8", "BuildSSDIndex")
    index.SetBuildParam("PostingPageLimit", "96", "BuildSSDIndex")
    index.SetBuildParam("NumberOfThreads", "16", "BuildSSDIndex")
    index.SetBuildParam("MaxCheck", "8192", "BuildSSDIndex")

    index.SetBuildParam("SearchPostingPageLimit", "96", "BuildSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "BuildSSDIndex")
    index.SetBuildParam("MaxDistRatio", "1000.0", "BuildSSDIndex")

    index.SetBuildParam("MaxCheck", "8192", "SearchSSDIndex")
    index.SetBuildParam("NumberOfThreads", "1", "SearchSSDIndex")
    index.SetBuildParam("SearchPostingPageLimit", "96", "SearchSSDIndex")
    index.SetBuildParam("SearchInternalResultNum", "64", "SearchSSDIndex")
    index.SetBuildParam("MaxDistRatio", "1000.0", "SearchSSDIndex")

    if index.Build(corpus, vector_number, False):
        index.Save(corpus_name + "_8")  # Save the index to the disk

    print("Build index accomplished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--corpus_path', type=str, default=None)
    args = parser.parse_args()

    build_index(args.name, args.corpus_path)
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import argparse
import numpy as np


def eval_latency(path):
    with open(path, "r", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        idx = 0
        temp = 0.0
        latency_list = []
        for [qid, latency] in tsvreader:
            idx += 1
            temp += float(latency)
            latency_list.append(float(latency))

        latency_numpy_list = np.array(latency_list)
        return temp/idx, np.percentile(latency_numpy_list, [50, 90, 99])


def eval_recall(gt_path, result_path):
    recall = 0.0
    with open(gt_path, "r", encoding="utf8") as f_gt, \
         open(result_path, "r", encoding="utf8") as f_result:
        gt_tsvreader = csv.reader(f_gt, delimiter="\t")
        result_tsvreader = csv.reader(f_result, delimiter="\t")
        
        gt_dict = {}
        for [gt_qid, gt_docid] in gt_tsvreader:
            if int(gt_qid) not in gt_dict.keys():
                gt_dict[int(gt_qid)] = [gt_docid]
            else:
                gt_dict[int(gt_qid)].append(gt_docid)
        
        result_dict = {}
        for [result_qid, result_docid, _, _] in result_tsvreader:
            if int(result_qid) not in result_dict.keys():
                result_dict[int(result_qid)] = [result_docid]
            else:
                result_dict[int(result_qid)].append(result_docid)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latency-path', type=str, default="",
                        help='latency file path')
    parser.add_argument('--gt-path', type=str, default="",
                        help='groundtruth file path')
    parser.add_argument('--qrels-path', type=str, default="",
                        help='query results file path')
    args = parser.parse_args()

    if (args.qrels_path != "") and (args.gt_path != ""):
        recall = eval_recall(args.gt_path, args.qrels_path)
        print(recall)
    if args.latency_path != "":
        avg_latency, latency_list = eval_latency(args.latency_path)
        print(avg_latency, latency_list)

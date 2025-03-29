import json
import os
from multiprocessing import Pool, cpu_count

import faiss
import numpy as np
import pytrec_eval
import torch
import gc

from tqdm import trange, tqdm
from typing import List, Dict, Tuple


def evaluate_mrr(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int]) -> Tuple[Dict[str, float]]:
    
    MRR = {}
    
    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]   
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])    
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"]/len(qrels), 5)

    return MRR

def search(queries_emb, doc_emb, topk: int = 100):
    gc.collect()
    torch.cuda.empty_cache()

    faiss_index = faiss.index_factory(doc_emb.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    doc_emb = doc_emb.astype(np.float32)
    faiss_index.train(doc_emb)
    faiss_index.add(doc_emb)

    dev_query_size = queries_emb.shape[0]
    all_scores = []
    all_indices = []
    for i in tqdm(range(0, dev_query_size, 32), desc="Searching"):
        j = min(i + 32, dev_query_size)
        query_embedding = queries_emb[i: j]
        score, indice = faiss_index.search(query_embedding.astype(np.float32), k=topk)
        all_scores.append(score)
        all_indices.append(indice)

    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    
    return all_scores, all_indices

def evaluate(metrics: List[str] = ['recall', 'mrr', 'ndcg'],
             k_values: List[int] = [1, 10],
             ground_truths: List[Dict] = None,
             predicts: List = None,
             scores: List = None):
    # labels = [[i] for i in range(len(queries))]
    # predicts = list([list(l) for l in all_indices])

    # ground_truths = {}
    # for i in range(len(labels)):
    #     ground_truths[str(i)] = {str(i): 1}

    retrieval_results = {}
    for i in range(len(predicts)):
        tmp = {}
        for j in range(len(predicts[0])):
            tmp[str(predicts[i][j])] = float(scores[i][j])
        retrieval_results[str(i)] = tmp

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"Precision@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(ground_truths,
                                               {map_string, ndcg_string, recall_string, precision_string})

    scores = evaluator.evaluate(retrieval_results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"Precision@{k}"] += scores[query_id]["P_" + str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"Precision@{k}"] = round(precision[f"Precision@{k}"] / len(scores), 5)

    mrr = evaluate_mrr(ground_truths, retrieval_results, k_values)

    data = {}

    if 'mrr' in metrics:
        data['mrr'] = mrr
    if 'recall' in metrics:
        data['recall'] = recall
    if 'ndcg' in metrics:
        data['ndcg'] = ndcg
    if 'map' in metrics:
        data['map'] = _map
    if 'precision' in metrics:
        data['precision'] = precision
    
    return data

def evaluate_better(metrics: List[str] = ['recall', 'mrr', 'ndcg'],
                    k_values: List[int] = [1, 10],
                    ground_truths: List[Dict] = None,
                    retrieval_results: List[Dict] = None):
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"Precision@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(ground_truths,
                                               {map_string, ndcg_string, recall_string, precision_string})

    scores = evaluator.evaluate(retrieval_results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"Precision@{k}"] += scores[query_id]["P_" + str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"Precision@{k}"] = round(precision[f"Precision@{k}"] / len(scores), 5)

    mrr = evaluate_mrr(ground_truths, retrieval_results, k_values)

    data = {}

    if 'mrr' in metrics:
        data['mrr'] = mrr
    if 'recall' in metrics:
        data['recall'] = recall
    if 'ndcg' in metrics:
        data['ndcg'] = ndcg
    if 'map' in metrics:
        data['map'] = _map
    if 'precision' in metrics:
        data['precision'] = precision
    
    return data
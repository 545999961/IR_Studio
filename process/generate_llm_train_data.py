import json
import multiprocessing
import numpy as np

from ir_studio.src.prompts import *
from ir_studio.src.evaluation.evaluate import search
from transformers import AutoModel
from typing import List
from tqdm import trange


def generate_llm_dpo_train_data(
    queries_corpus_list: List[List[dict]] = None,
    search_dtype: str = 'answer',
    result_dtype: str = 'passage',
    retrieval_model: AutoModel = None,
    threshold: float = 0.95,
    batch_size: int = 512,
    max_length: int = 1024,
    corpus: List[str] = None,
    dataset_name: str = None,
    etype: str = 'answer',
    use_rule1: bool = True
):
    data = []

    queries_list = []
    corpus = []
    raw_queries = []
    for qc in queries_corpus_list:
        raw_queries = [d['query'] for d in qc]
        if 'new_query' in qc[0].keys():
            queries_list.append([d['new_query'] for d in qc])
        else:
            queries_list.append([d[search_dtype] for d in qc])
        corpus = [d[result_dtype] for d in qc]
        
    doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
    raw_queries_emb = retrieval_model.encode_queries(raw_queries, batch_size=batch_size, max_length=max_length)
    raw_scores = np.einsum('ij,ij->i', raw_queries_emb, doc_emb)
    all_scores_list = []
    for queries in queries_list:
        # queries = ['Generate the topic about this passage: ' + q for q in queries]
        queries_emb = raw_queries_emb * 0.8 + retrieval_model.encode_queries(queries, batch_size=batch_size, max_length=max_length, etype=etype) * 0.2
        # queries_emb = raw_queries_emb
        all_scores_list.append(np.einsum('ij,ij->i', queries_emb, doc_emb))
    
    for i in range(len(all_scores_list[0])):
        raw_score = raw_scores[i]
        all_scores = [e[i] for e in all_scores_list]
        items = [(idx, all_scores[idx]) for idx in range(len(all_scores))]
        sorted_idx = [idx for idx, _ in sorted(items, key=lambda x: x[1], reverse=False)]
        min_score = max(all_scores)
        for idx in sorted_idx:
            if abs(1 - all_scores[idx] / raw_score) < 0.1:
                min_score = all_scores[idx]
                break
        min_score = min(all_scores)
        max_score = max(all_scores)
        # min_score = min(all_scores)
        # import time
        # print('max score:', max_score, ';', 'min_score:', min_score, ';', 'raw_score:', raw_score, ';', 
        #       'max_ans_score:', max_score - raw_score * 0.8, ';', 'min_ans_score:', min_score - raw_score * 0.8, ';',
        #       'max_ans_score * 0.95:', (max_score - raw_score * 0.8) * 0.95)
        # time.sleep(1)
        if use_rule1:
            if max_score > raw_score and (max_score - raw_score * 0.8) * threshold >= (min_score - raw_score * 0.8):
                # print('use')
                tmp = {
                    'prompt': queries_corpus_list[0][i]['query'],
                    'chosen': queries_corpus_list[all_scores.index(max_score)][i][search_dtype],
                    'rejected': queries_corpus_list[all_scores.index(min_score)][i][search_dtype],
                }
                tmp['chosen_score'] = float(max_score / raw_score)
                tmp['rejected_score'] = float(min_score / raw_score)
                data.append(tmp)
        else:
            if (max_score - raw_score * 0.8) * threshold >= (min_score - raw_score * 0.8):
                # print('use')
                tmp = {
                    'prompt': queries_corpus_list[0][i]['query'],
                    'chosen': queries_corpus_list[all_scores.index(max_score)][i][search_dtype],
                    'rejected': queries_corpus_list[all_scores.index(min_score)][i][search_dtype],
                }
                tmp['chosen_score'] = float(max_score / raw_score)
                tmp['rejected_score'] = float(min_score / raw_score)
                data.append(tmp)

    return data
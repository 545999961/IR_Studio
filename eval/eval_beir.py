import argparse
import os
import json
import time
import random
import shutil
import torch

import pandas as pd
import numpy as np

from ir_studio.src.prompts import *
from ir_studio.src.evaluation import evaluate, search, evaluate_better
from FlagEmbedding import FlagModel
from tqdm import tqdm
from typing import List


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--api_key', type=str, default="sk-g7Tco5jijDw0h9QJAfA2A1Af9d95464f9a828fB1B5F845D9")
    parser.add_argument('--base_url', type=str, default="https://api.xiaoai.plus/v1")
    parser.add_argument('--generate_model_path', type=str, default="/share/chaofan/models/Meta-Llama-3-8B")
    parser.add_argument('--generate_model_lora_path', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--model_type', type=str, default="llm")
    parser.add_argument('--retrieval_model_name', type=str, default="/share/chaofan/models/models/bge-large-en-v1.5")
    parser.add_argument('--pooling_method', type=str, default='cls')
    parser.add_argument('--retrieval_query_prompt', type=str,
                        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--query_ratio', type=float, default=1.0)
    parser.add_argument('--answer_ratio', type=float, default=1.0)
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--query_output_dir', type=str, default="./output")
    parser.add_argument('--corpus_output_dir', type=str, default=None)
    parser.add_argument('--result_output_path', type=str, default="./result.json")
    parser.add_argument('--metrics', type=List[str], default=['recall', 'mrr', 'ndcg'])
    parser.add_argument('--eval_type', type=str, default='combine')
    parser.add_argument('--generate_number', type=int, default=1)
    parser.add_argument('--eval_weight', type=float, default=1.0)
    parser.add_argument('--doc_emb_path', type=str, default=None)
    parser.add_argument('--k_values', type=List[int], default=[10])
    parser.add_argument('--queries_num', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--linear_path', type=str, default=None)
    parser.add_argument('--answer_type', type=str, default='answer')
    parser.add_argument('--normalize_embeddings', type=str, default='False')

    opt = parser.parse_args()

    return opt


def main(opt):
    api_key = opt.api_key
    base_url = opt.base_url
    generate_model_path = opt.generate_model_path
    generate_model_lora_path = opt.generate_model_lora_path
    temperature = opt.temperature
    gpu_memory_utilization = opt.gpu_memory_utilization
    tensor_parallel_size = opt.tensor_parallel_size
    top_p = opt.top_p
    max_tokens = opt.max_tokens
    model_type = opt.model_type
    retrieval_model_name = opt.retrieval_model_name
    retrieval_query_prompt = opt.retrieval_query_prompt
    dataset_path = opt.dataset_path
    query_output_dir = opt.query_output_dir
    corpus_output_dir = opt.corpus_output_dir
    result_output_path = opt.result_output_path
    max_length = opt.max_length
    batch_size = opt.batch_size
    metrics = opt.metrics
    k_values = opt.k_values
    eval_type = opt.eval_type
    eval_weight = opt.eval_weight
    doc_emb_path = opt.doc_emb_path
    queries_num = opt.queries_num
    pooling_method = opt.pooling_method
    dataset_name = opt.dataset_name
    generate_number = opt.generate_number
    query_ratio = opt.query_ratio
    answer_ratio = opt.answer_ratio
    linear_path = opt.linear_path
    answer_type = opt.answer_type
    normalize_embeddings = opt.normalize_embeddings
    if normalize_embeddings == 'False':
        normalize_embeddings = False
    else:
        normalize_embeddings = True

    print('normalize embeddings:', normalize_embeddings)
    print('pooling method:', opt.pooling_method)

    if eval_type == 'combine':
        generate_number = 1

    try:
        os.makedirs('/'.join(result_output_path.split('/')[:-1]), exist_ok=True)
    except:
        pass

    seed = str(time.time() + random.random())
    prompts_input_dir = os.path.join('tmp_path', seed, 'input')
    prompts_output_dir = os.path.join('tmp_path', seed, 'output')
    os.makedirs(prompts_input_dir, exist_ok=True)
    os.makedirs(prompts_output_dir, exist_ok=True)

    if generate_model_lora_path is not None and not os.path.exists(generate_model_lora_path):
        generate_model_lora_path = None

    input_dirs = ['', '/cqadupstack']
    generate_flag = False
    for input_dir in input_dirs:
        for file in os.listdir('/share/chaofan/dataset/beir_all' + input_dir):
            if dataset_name is not None:
                if input_dir == '' and file not in dataset_name:
                    continue
                if input_dir == '/cqadupstack' and 'cqadupstack' not in dataset_name:
                    continue
                if file == 'fever' and 'fever' in dataset_name:
                    if 'climate-fever' in dataset_name:
                        if dataset_name.count('fever') == 1:
                            continue

            if file == 'cqadupstack':
                continue
            os.makedirs(os.path.join(query_output_dir + input_dir, file), exist_ok=True)
            if corpus_output_dir is not None:
                os.makedirs(os.path.join(corpus_output_dir + input_dir, file), exist_ok=True)
            query_output_path = os.path.join(query_output_dir + input_dir, file, 'queries.json')
            if corpus_output_dir is not None:
                corpus_output_path = os.path.join(corpus_output_dir + input_dir, file, 'corpus.npy')

            corpus_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'corpus.jsonl')
            queries_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'queries.jsonl')
            if 'msmarco' not in file:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'test' + '.tsv')
            else:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'dev' + '.tsv')

            all_queries = {}
            with open(queries_path) as f:
                for line in f:
                    tmp = json.loads(line)
                    all_queries[str(tmp['_id'])] = tmp['text']

            rels = pd.read_csv(rels_path, delimiter='\t')
            queries = []
            queries_ids = []
            for i in range(len(rels)):
                if str(rels['query-id'][i]) not in queries_ids:
                    queries_ids.append(str(rels['query-id'][i]))

            for query_id in queries_ids:
                queries.append(all_queries[query_id])

            if os.path.exists(query_output_path):
                if model_type in ['gpt', 'llm', 'llm_instruct']:
                    queries = json.load(open(query_output_path))
            else:
                if model_type in ['gpt', 'llm', 'llm_instruct']:
                    if input_dir == '':
                        prompts = [get_additional_info_generation_prompt(file, q) for q in queries]
                        tf = file
                    else:
                        prompts = [get_additional_info_generation_prompt('cqadupstack', q) for q in queries]
                        tf = 'cqadupstack-' + file

                    for j in range(generate_number):
                        with open(os.path.join(prompts_input_dir, f'{tf}-{j}.json'), 'w') as f:
                            json.dump(prompts, f)

                    generate_flag = True

    ### run bash to call LLM
    devices = torch.cuda.device_count()
    if generate_flag is True:
        cmd = f"python -m ir_studio.multi \
                --generate_model_path {generate_model_path} \
                --generate_model_lora_path {generate_model_lora_path} \
                --temperature {temperature} \
                --gpu_memory_utilization {gpu_memory_utilization} \
                --top_p {top_p} \
                --max_tokens {max_tokens} \
                --model_type {model_type} \
                --input_dir {prompts_input_dir} \
                --output_dir {prompts_output_dir} \
                --num_gpus {devices} \
                --rm_tmp True "
        os.system(cmd)

    for input_dir in input_dirs:
        for file in os.listdir('/share/chaofan/dataset/beir_all' + input_dir):
            if dataset_name is not None:
                if input_dir == '' and file not in dataset_name:
                    continue
                if input_dir == '/cqadupstack' and 'cqadupstack' not in dataset_name:
                    continue
                if file == 'fever' and 'fever' in dataset_name:
                    if 'climate-fever' in dataset_name:
                        if dataset_name.count('fever') == 1:
                            continue

            if file == 'cqadupstack':
                continue
            os.makedirs(os.path.join(query_output_dir + input_dir, file), exist_ok=True)
            if corpus_output_dir is not None:
                os.makedirs(os.path.join(corpus_output_dir + input_dir, file), exist_ok=True)
            query_output_path = os.path.join(query_output_dir + input_dir, file, 'queries.json')
            if corpus_output_dir is not None:
                corpus_output_path = os.path.join(corpus_output_dir + input_dir, file, 'corpus.npy')

            corpus_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'corpus.jsonl')
            queries_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'queries.jsonl')
            if 'msmarco' not in file:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'test' + '.tsv')
            else:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'dev' + '.tsv')

            all_queries = {}
            with open(queries_path) as f:
                for line in f:
                    tmp = json.loads(line)
                    all_queries[str(tmp['_id'])] = tmp['text']

            rels = pd.read_csv(rels_path, delimiter='\t')
            queries = []
            queries_ids = []
            for i in range(len(rels)):
                if str(rels['query-id'][i]) not in queries_ids:
                    queries_ids.append(str(rels['query-id'][i]))

            for query_id in queries_ids:
                queries.append(all_queries[query_id])

            if os.path.exists(query_output_path):
                if model_type in ['gpt', 'llm', 'llm_instruct']:
                    new_queries = json.load(open(query_output_path))
                else:
                    new_queries = queries
            else:
                if model_type in ['gpt', 'llm', 'llm_instruct']:
                    if input_dir == '':
                        tf = file
                    else:
                        tf = 'cqadupstack-' + file

                    if eval_type == 'combine':
                        new_queries = []
                    elif eval_type == 'sep' or eval_type == 'sep_wo_prefix':
                        new_queries = [[] for _ in range(generate_number + 1)]
                        new_queries[0] = queries
                    elif eval_type == 'sep_answer':
                        new_queries = [[] for _ in range(generate_number + 1)]
                        new_queries[0] = queries

                    for idx in range(generate_number):
                        answers = json.load(open(os.path.join(prompts_output_dir, f'{tf}-{idx}.json')))

                        # print(answers)

                        for i in range(len(answers)):
                            if input_dir == '':
                                instruction = TASK_DICT[file]
                            else:
                                instruction = TASK_DICT['cqadupstack']
                            query = queries[i]
                            answer = answers[i]
                            if eval_type == 'combine':
                                # new_queries.append("Instruction: {instruction}\nQuery: {query}\nPossible response: {answer}".format(instruction=instruction, query=query, answer=answer))
                                new_queries.append(
                                    "{query}\n{answer}".format(instruction=instruction, query=query, answer=answer))
                            elif eval_type == 'sep':
                                new_queries[idx + 1].append('Generate the topic about this passage: ' + answer)
                                # new_queries[idx + 1].append(answer)
                            elif eval_type == 'sep_wo_prefix':
                                new_queries[idx + 1].append(answer)
                            elif eval_type == 'sep_answer':
                                new_queries[idx + 1].append("{query}\n{answer}".format(query=query, answer=answer))
                else:
                    new_queries = queries

                with open(query_output_path, 'w') as f:
                    json.dump(new_queries, f)

    if os.path.exists(os.path.join('tmp_path', seed)):
        shutil.rmtree(os.path.join('tmp_path', seed))

    retrieval_model = FlagModel(retrieval_model_name,
                                query_instruction_for_retrieval=retrieval_query_prompt,
                                pooling_method=pooling_method,
                                use_fp16=True,
                                linear_path=linear_path,
                                normalize_embeddings=normalize_embeddings
                                # acceleration_type='multi_process'
                                )
    results = {}
    if os.path.exists(result_output_path):
        results = json.load(open(result_output_path))

    for input_dir in input_dirs:
        for file in tqdm(os.listdir('/share/chaofan/dataset/beir_all' + input_dir)):
            if dataset_name is not None:
                if input_dir == '' and file not in dataset_name:
                    continue
                if input_dir == '/cqadupstack' and 'cqadupstack' not in dataset_name:
                    continue
                if file == 'fever' and 'fever' in dataset_name:
                    if 'climate-fever' in dataset_name:
                        if dataset_name.count('fever') == 1:
                            continue

            if input_dir == '':
                if file in results.keys():
                    continue
            else:
                if "cqa.{}".format(file) in results.keys():
                    continue

            if file == 'cqadupstack':
                continue
            os.makedirs(os.path.join(query_output_dir + input_dir, file), exist_ok=True)
            if corpus_output_dir is not None:
                os.makedirs(os.path.join(corpus_output_dir + input_dir, file), exist_ok=True)
            query_output_path = os.path.join(query_output_dir + input_dir, file, 'queries.json')
            if corpus_output_dir is not None:
                corpus_output_path = os.path.join(corpus_output_dir + input_dir, file, 'corpus.npy')
            # if file == 'fever' and os.path.exists(os.path.join(corpus_output_dir + input_dir, 'climate-fever', 'corpus.npy')):
            #     corpus_output_path = os.path.join(corpus_output_dir + input_dir, 'climate-fever', 'corpus.npy')
            # if file == 'climate-fever' and os.path.exists(os.path.join(corpus_output_dir + input_dir, 'fever', 'corpus.npy')):
            #     corpus_output_path = os.path.join(corpus_output_dir + input_dir, 'fever', 'corpus.npy')

            corpus_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'corpus.jsonl')
            queries_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'queries.jsonl')
            if 'msmarco' not in file:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'test' + '.tsv')
            else:
                rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels', 'dev' + '.tsv')

            corpus = []
            corpus_ids = []
            with open(corpus_path) as f:
                for line in f:
                    tmp = json.loads(line)
                    if tmp.get('title') == '':
                        corpus.append(tmp['text'])
                    else:
                        corpus.append((tmp['title'] + ' ' + tmp['text']).strip())
                    corpus_ids.append(str(tmp['_id']))

            all_queries = {}
            with open(queries_path) as f:
                for line in f:
                    tmp = json.loads(line)
                    all_queries[str(tmp['_id'])] = tmp['text']

            rels = pd.read_csv(rels_path, delimiter='\t')
            queries = []
            queries_ids = []
            for i in range(len(rels)):
                if str(rels['query-id'][i]) not in queries_ids:
                    queries_ids.append(str(rels['query-id'][i]))

            for query_id in queries_ids:
                queries.append(all_queries[query_id])

            queries = json.load(open(query_output_path))
            if eval_type == 'sep_wo_prefix':
                for i in range(len(queries)):
                    queries[i] = [e.replace('Generate the topic about this passage: ', '') for e in queries[i]]

            if corpus_output_dir is not None:
                if os.path.exists(corpus_output_path):
                    corpus_emb = np.load(corpus_output_path)
                else:
                    corpus_emb = retrieval_model.encode_corpus(corpus, batch_size=512)
                    np.save(corpus_output_path, corpus_emb)
            else:
                corpus_emb = retrieval_model.encode_corpus(corpus, batch_size=512)
            if isinstance(queries[0], list):
                queries_emb = retrieval_model.encode_queries(queries[0], batch_size=512) * query_ratio
                for tmp_queries in queries[1:]:
                    queries_emb += retrieval_model.encode_queries(tmp_queries, batch_size=512,
                                                                  etype=answer_type) * answer_ratio
            else:
                queries_emb = retrieval_model.encode_queries(queries, batch_size=512)
            scores, indices = search(queries_emb, corpus_emb)

            retrieval_results = {}
            idx = 0
            for score, indice in zip(scores, indices):
                indice = indice.tolist()
                tmp = {}
                for s, i in zip(score, indice):
                    if corpus_ids[i] != queries_ids[idx]:
                        tmp[str(i)] = float(round(s, 4))
                retrieval_results[str(idx)] = tmp
                idx += 1

            rels_path = os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'rels.json')
            if not os.path.exists(rels_path):
                ground_truths = {}
                with open(os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'qrels/test.tsv')) as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        t = line.strip().split('\t')
                        if ground_truths.get(queries_ids.index(t[0])) is None:
                            try:
                                ground_truths[queries_ids.index(t[0])] = {corpus_ids.index(t[1]): t[2]}
                            except:
                                ground_truths[queries_ids.index(t[0])] = {len(corpus_ids) + 1: t[2]}
                        else:
                            try:
                                ground_truths[queries_ids.index(t[0])][corpus_ids.index(t[1])] = t[2]
                            except:
                                ground_truths[queries_ids.index(t[0])][len(corpus_ids) + 1] = t[2]

                with open(rels_path, 'w') as f:
                    json.dump(ground_truths, f)

            tmp_ground_truths = json.load(
                open(os.path.join('/share/chaofan/dataset/beir_all' + input_dir, file, 'rels.json')))
            tmp_ground_truths = {str(k): {str(k2): int(v2) for k2, v2 in v.items()} for k, v in
                                 tmp_ground_truths.items()}

            result = evaluate_better(metrics=metrics,
                                     k_values=[10],
                                     ground_truths=tmp_ground_truths,
                                     retrieval_results=retrieval_results)
            if input_dir == '':
                results[file] = result
            else:
                results["cqa.{}".format(file)] = result

            with open(result_output_path, 'w') as f:
                json.dump(results, f)

    new_results = {
        'ndcg@10': 0.0,
        'recall@10': 0.0,
        'mrr@10': 0.0
    }
    cqa_results = {
        'ndcg@10': 0.0,
        'recall@10': 0.0,
        'mrr@10': 0.0
    }
    cqa_num = 0
    all_num = 0

    for r, v in results.items():
        if 'ndcg' not in v:
            continue
        if 'cqa.' in r:
            cqa_results['ndcg@10'] += v['ndcg']['NDCG@10']
            cqa_results['recall@10'] += v['recall']['Recall@10']
            cqa_results['mrr@10'] += v['mrr']['MRR@10']
            cqa_num += 1
        else:
            new_results['ndcg@10'] += v['ndcg']['NDCG@10']
            new_results['recall@10'] += v['recall']['Recall@10']
            new_results['mrr@10'] += v['mrr']['MRR@10']
            all_num += 1

    if cqa_num > 0:
        all_num += 1

        results['cqa'] = {
            "mrr": {
                "MRR@10": cqa_results['mrr@10'] / cqa_num
            },
            "recall": {
                "Recall@10": cqa_results['recall@10'] / cqa_num
            },
            "ndcg": {
                "NDCG@10": cqa_results['ndcg@10'] / cqa_num
            }
        }

        results['avg'] = {
            'ndcg@10': (new_results['ndcg@10'] + cqa_results['ndcg@10'] / cqa_num) / all_num,
            'recall@10': (new_results['recall@10'] + cqa_results['recall@10'] / cqa_num) / all_num,
            'mrr@10': (new_results['mrr@10'] + cqa_results['mrr@10'] / cqa_num) / all_num
        }

    else:
        results['avg'] = {
            'ndcg@10': new_results['ndcg@10'] / all_num,
            'recall@10': new_results['recall@10'] / all_num,
            'mrr@10': new_results['mrr@10'] / all_num
        }

    with open(result_output_path, 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
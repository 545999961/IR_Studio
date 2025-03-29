import argparse
import os
import json
import sys
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
from typing import List, Any, Dict, Optional, Union

from mteb.evaluation.evaluators import RetrievalEvaluator
from air_benchmark import Retriever, Reranker, AIRBench


class NewRetriever(FlagModel):
    def __init__(
            self,
            generate_model_path,
            generate_model_lora_path,
            temperature,
            gpu_memory_utilization,
            top_p,
            max_tokens,
            model_type,
            query_output_dir,
            emb_save_dir,
            dataset_name,
            generate_number,
            eval_type,
            query_ratio,
            answer_ratio,
            **kwargs
    ):
        self.generate_model_path = generate_model_path
        self.generate_model_lora_path = generate_model_lora_path
        self.temperature = temperature
        self.gpu_memory_utilization = gpu_memory_utilization
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model_type = model_type
        self.query_output_dir = query_output_dir
        self.emb_save_dir = emb_save_dir
        self.dataset_name = dataset_name
        self.generate_number = generate_number
        self.eval_type = eval_type

        self.query_ratio = query_ratio
        self.answer_ratio = answer_ratio

        if self.generate_model_lora_path is not None and not os.path.exists(self.generate_model_lora_path):
            self.generate_model_lora_path = None

        super().__init__(**kwargs)

    def encode_queries(
            self,
            queries: Union[List[str], str],
            prompt_name: str = None,
            batch_size: int = 128,
            show_progress_bar: bool = True,
            convert_to_tensor: bool = True,
            **kwargs: Any
    ):
        for k in kwargs.keys():
            print(k, ':', kwargs[k])
        seed = str(time.time() + random.random())
        prompts_input_dir = os.path.join('tmp_path', seed, 'input')
        prompts_output_dir = os.path.join('tmp_path', seed, 'output')
        os.makedirs(prompts_input_dir, exist_ok=True)
        os.makedirs(prompts_output_dir, exist_ok=True)

        query_output_path = os.path.join(self.query_output_dir, self.dataset_name, 'queries.json')
        os.makedirs(os.path.join(self.query_output_dir, self.dataset_name), exist_ok=True)

        generate_flag = False
        if os.path.exists(query_output_path):
            if self.model_type in ['gpt', 'llm', 'llm_instruct']:
                queries = json.load(open(query_output_path))
        else:
            if self.model_type in ['gpt', 'llm', 'llm_instruct']:
                prompts = [get_additional_info_generation_prompt(self.dataset_name, q) for q in queries]
                tf = self.dataset_name

                for j in range(self.generate_number):
                    with open(os.path.join(prompts_input_dir, f'{tf}-{j}.json'), 'w') as f:
                        json.dump(prompts, f)

                generate_flag = True

        generate_devices = torch.cuda.device_count()
        if generate_flag is True:
            cmd = f"python -m ir_studio.multi \
                        --generate_model_path {self.generate_model_path} \
                        --generate_model_lora_path {self.generate_model_lora_path} \
                        --temperature {self.temperature} \
                        --gpu_memory_utilization {self.gpu_memory_utilization} \
                        --top_p {self.top_p} \
                        --max_tokens {self.max_tokens} \
                        --model_type {self.model_type} \
                        --input_dir {prompts_input_dir} \
                        --output_dir {prompts_output_dir} \
                        --num_gpus {generate_devices} \
                        --rm_tmp True "
            os.system(cmd)

        if os.path.exists(query_output_path):
            if self.model_type in ['gpt', 'llm', 'llm_instruct']:
                new_queries = json.load(open(query_output_path))
                if self.eval_type == 'sep_wo_prefix':
                    for i in range(len(new_queries)):
                        new_queries[i] = [e.replace('Generate the topic about this passage: ', '') for e in new_queries[i]]
            else:
                new_queries = queries
        else:
            if self.model_type in ['gpt', 'llm', 'llm_instruct']:
                tf = self.dataset_name

                if self.eval_type == 'combine':
                    new_queries = []
                elif self.eval_type == 'sep' or  self.eval_type == 'sep_wo_prefix':
                    new_queries = [[] for _ in range(self.generate_number + 1)]
                    new_queries[0] = queries
                elif self.eval_type == 'sep_answer':
                    new_queries = [[] for _ in range(self.generate_number + 1)]
                    new_queries[0] = queries

                for idx in range(self.generate_number):
                    answers = json.load(open(os.path.join(prompts_output_dir, f'{tf}-{idx}.json')))

                    for i in range(len(answers)):
                        instruction = TASK_DICT[self.dataset_name]
                        query = queries[i]
                        answer = answers[i]
                        if self.eval_type == 'combine':
                            # new_queries.append("Instruction: {instruction}\nQuery: {query}\nPossible response: {answer}".format(instruction=instruction, query=query, answer=answer))
                            new_queries.append(
                                "{query}\n{answer}".format(instruction=instruction, query=query, answer=answer))
                        elif self.eval_type == 'sep':
                            new_queries[idx + 1].append('Generate the topic about this passage: ' + answer)
                        elif self.eval_type == 'sep_wo_prefix':
                            new_queries[idx + 1].append(answer)
                            # new_queries[idx + 1].append(answer)
                        elif self.eval_type == 'sep_answer':
                            new_queries[idx + 1].append("{query}\n{answer}".format(query=query, answer=answer))
            else:
                new_queries = queries
            with open(query_output_path, 'w') as f:
                json.dump(new_queries, f)

        if os.path.exists(os.path.join('tmp_path', seed)):
            shutil.rmtree(os.path.join('tmp_path', seed))

        if isinstance(new_queries[0], list):
            queries_emb = super().encode_queries(new_queries[0], **kwargs) * self.query_ratio
            for tmp_queries in new_queries[1:]:
                queries_emb += super().encode_queries(tmp_queries, **kwargs) * self.answer_ratio
        else:
            queries_emb = super().encode_queries(new_queries, **kwargs)

        return queries_emb

    def encode_corpus(
            self,
            sentences: List[str],
            prompt_name: str = None,
            request_qid: str = None,
            batch_size: int = 128,
            show_progress_bar: bool = True,
            convert_to_tensor: bool = True,
            **kwargs: Any
    ):
        # for k in kwargs.keys():
        #     print('encode corpus', k, ':', kwargs[k])
        print('=========================================encode corpus=======================================')

        if not isinstance(sentences[0], str):
            sentences = [e['text'] for e in sentences]

        corpus_emb_path = os.path.join(self.emb_save_dir, self.dataset_name, 'corpus.npy')
        os.makedirs(os.path.join(self.emb_save_dir, self.dataset_name), exist_ok=True)
        if os.path.exists(corpus_emb_path):
            return np.load(corpus_emb_path)
        else:
            corpus_emb = super().encode_corpus(sentences, **kwargs)
            np.save(corpus_emb_path, corpus_emb)
            return corpus_emb


class EmbeddingModelRetriever(Retriever):
    def __init__(self, embedding_model, search_top_k: int = 1000, **kwargs):
        super().__init__(search_top_k)
        self.embedding_model = embedding_model
        self.retriever = RetrievalEvaluator(
            retriever=embedding_model,
            k_values=[self.search_top_k],
            score_function='dot',
            **kwargs,
        )

    def __str__(self):
        return 'test'

    def __call__(
            self,
            corpus: Dict[str, Dict[str, Any]],
            queries: Dict[str, str],
            **kwargs,
    ):
        search_results = self.retriever(corpus=corpus, queries=queries)
        return search_results


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--generate_model_path', type=str, default="/share/chaofan/models/Meta-Llama-3-8B")
    parser.add_argument('--generate_model_lora_path', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--model_type', type=str, default="llm")
    parser.add_argument('--retrieval_model_name', type=str, default="/share/chaofan/models/models/bge-large-en-v1.5")
    parser.add_argument('--retrieval_query_prompt', type=str,
                        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=768)
    parser.add_argument('--eval_type', type=str, default="sep")
    parser.add_argument('--query_ratio', type=float, default=1.0)
    parser.add_argument('--answer_ratio', type=float, default=1.0)
    parser.add_argument('--queries_num', type=int, default=1)
    parser.add_argument('--normalize_embeddings', type=str, default='False')
    parser.add_argument('--pooling_method', type=str, default='cls')

    parser.add_argument('--query_output_dir', type=str, default="./output")
    parser.add_argument('--emb_save_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--result_output_path', type=str, default="./result.json")
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--task_types', type=str, default='qa')
    parser.add_argument('--domains', type=str, default=None)

    opt = parser.parse_args()

    return opt


def main(opt):
    generate_model_path = opt.generate_model_path
    generate_model_lora_path = opt.generate_model_lora_path
    temperature = opt.temperature
    gpu_memory_utilization = opt.gpu_memory_utilization
    top_p = opt.top_p
    max_tokens = opt.max_tokens
    model_type = opt.model_type
    retrieval_model_name = opt.retrieval_model_name
    retrieval_query_prompt = opt.retrieval_query_prompt
    query_output_dir = opt.query_output_dir
    output_dir = opt.output_dir
    emb_save_dir = opt.emb_save_dir
    result_output_path = opt.result_output_path
    max_length = opt.max_length
    batch_size = opt.batch_size
    eval_type = opt.eval_type
    pooling_method = opt.pooling_method
    dataset_name = opt.dataset_name
    generate_number = opt.queries_num
    query_ratio = opt.query_ratio
    answer_ratio = opt.answer_ratio
    normalize_embeddings = opt.normalize_embeddings
    task_types = opt.task_types.split()
    domains = opt.domains.split()
    if normalize_embeddings == 'False':
        normalize_embeddings = False
    else:
        normalize_embeddings = True

    if os.path.exists(result_output_path):
        print(json.load(open(result_output_path)))
        sys.exit(0)

    print('normalize embeddings:', normalize_embeddings)
    print('pooling method:', opt.pooling_method)

    embedding_model = NewRetriever(
        generate_model_path=generate_model_path,
        generate_model_lora_path=generate_model_lora_path,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        top_p=top_p,
        max_tokens=max_tokens,
        model_type=model_type,
        query_output_dir=query_output_dir,
        emb_save_dir=emb_save_dir,
        dataset_name=dataset_name,
        generate_number=generate_number,
        eval_type=eval_type,
        query_ratio=query_ratio,
        answer_ratio=answer_ratio,
        # for init
        model_name_or_path=retrieval_model_name,
        normalize_embeddings=normalize_embeddings,
        pooling_method=pooling_method,
        use_fp16=True,
        query_instruction_for_retrieval=retrieval_query_prompt,
        max_query_length=max_length,
        max_passage_length=max_length,
        batch_size=batch_size,
    )

    retriever = EmbeddingModelRetriever(
        embedding_model,
        search_top_k=1000,
        corpus_chunk_size=9999999999999,
    )

    evaluation = AIRBench(
        benchmark_version='AIR-Bench_24.05',
        task_types=task_types,
        domains=domains,
        languages='en',
        splits=['dev'],
        cache_dir='/share/chaofan/cache/air_data_cache',
    )

    evaluation.run(
        retriever,
        reranker=None,
        output_dir=output_dir,
        overwrite=False,
    )

    # os.makedirs(os.path.dirname(result_output_path), exist_ok=True)

    AIRBench.evaluate_dev(
        benchmark_version='AIR-Bench_24.05',
        search_results_save_dir=output_dir,
        cache_dir='/share/chaofan/cache/air_data_cache',
        output_method='json',
        output_path=result_output_path,
        metrics=['ndcg_at_10'],
    )

    shutil.rmtree(output_dir)
    result = json.load(open(result_output_path))
    result = result['test']['NoReranker'][task_types[0]][domains[0]]['en']['default']
    metrics = list(result.keys())
    metrics = list(set([e.split('_')[0] for e in metrics]))
    k_value = list(result.keys())
    k_value = list(set([e.split('_')[-1] for e in k_value]))
    new_result = {}
    for metric in metrics:
        new_result[metric] = {}
        k_value = [10]
        for k in k_value:
            if metric == 'recall':
                new_result[metric][f"Recall@{k}"] = result[f'{metric}_at_{k}']
            else:
                new_result[metric][f"{metric.upper()}@{k}"] = result[f'{metric}_at_{k}']
    new_result["avg"] = {
        "ndcg@10": new_result['ndcg']['NDCG@10'],
        "recall@10": new_result['recall']['Recall@10'],
        "mrr@10": new_result['mrr']['MRR@10']
    }
    print(new_result['avg'])
    with open(result_output_path, 'w') as f:
        json.dump(new_result, f, indent=4)


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
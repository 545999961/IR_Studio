import torch
import copy
import time
import json
import re
import numpy as np
import subprocess

from typing import List
from tqdm import trange, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from FlagEmbedding import FlagReranker
from ir_studio.src.generation_agent import GPTAgent, LLMAgent, LLMInstructAgent

from logits_modeling import LogitsModel
from logits_modeling_yes import YesLogitsModel

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    numbers = [int(num) for num in numbers]
    return numbers

def get_distill_data(
    model_name_or_path: str = None,
    model_type: str = 'llm_instruct',
    train_data: List = None,
    prompts: List[str] = None,
    batch_size: int = 32,
):
    # command = [
    #     'vllm', 'serve',
    #     '/share/chaofan/models/Qwen2.5-72B-Instruct',
    #     '--served-model-name', 'Qwen-2-5-72B-Instruct',
    #     '--max-model-len', '32768',
    #     '--tensor-parallel-size', '8',
    #     '--gpu_memory_utilization', '0.85'
    # ]

    # process = subprocess.Popen(command)
    ns = time.time()
    subprocess.Popen(
        f"nohup bash /share/chaofan/code/IR-Studio-up/script/1230-contriever/vllm_log/start_vllm.sh > /share/chaofan/code/IR-Studio-up/script/1230-contriever/vllm_log/start_vllm_{ns}.txt 2>&1 &",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(20)

    llm_for_rank = GPTAgent(
        model_name='Qwen-2-5-72B-Instruct',
        # model_name='Llama-3-1-70B-Instruct',
        api_key='empty',
        base_url='http://localhost:8000/v1/',
        temperature=0
    )

    # generated_rank_results = []
    # for i in range(0, len(prompts), 100):
    #     generated_rank_results.extend(llm_for_rank.generate(prompts[i: i + 100]))
    #     print(generated_rank_results[i: i + 100])
    
    generated_rank_results = llm_for_rank.generate(prompts)

    # try:
    #     fis = '/share/chaofan/code/IR-Studio-up/code/process/train_accelerate/' + str(time.time()) + '.json'
    #     with open(fis, 'w') as f:
    #         json.dump(generated_rank_results, f)
    # except:
    #     pass

    # generated_rank_results = json.load(open('/share/chaofan/code/IR-Studio-up/code/process/train_accelerate/1735301231.2612002.json'))

    process = subprocess.Popen(['pkill', '-f', 'vllm'])
    
    ps = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    grep = subprocess.Popen(['grep', 'vllm'], stdin=ps.stdout, stdout=subprocess.PIPE)
    ps.stdout.close()
    process_list = grep.communicate()[0].decode('utf-8').strip().split('\n')

    for line in process_list:
        if 'grep' not in line:  # 排除 grep 本身的进程
            try:
                pid = int(line.split()[1])  # 获取 PID
                subprocess.run(['kill', str(pid)])  # 杀掉进程
                print(f"Killed process {pid}")
            except Exception as e:
                print(f"Error: {e}")
    
    ps = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
    grep = subprocess.Popen(['grep', 'multiprocessing.spawn'], stdin=ps.stdout, stdout=subprocess.PIPE)
    ps.stdout.close()
    process_list = grep.communicate()[0].decode('utf-8').strip().split('\n')

    for line in process_list:
        if 'grep' not in line:  # 排除 grep 本身的进程
            try:
                pid = int(line.split()[1])  # 获取 PID
                subprocess.run(['kill', str(pid)])  # 杀掉进程
                print(f"Killed process {pid}")
            except Exception as e:
                print(f"Error: {e}")
    
    subprocess.Popen(
        f"rm /share/chaofan/code/IR-Studio-up/script/1230-contriever/vllm_log/start_vllm_{ns}.txt",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    idx = 0
    for d, res in zip(train_data, generated_rank_results):
        # res = res.split('>')
        # res = [int(i.strip().strip('[').strip(']')) for i in res]
        res = extract_numbers(res)
        passages = []
        passages.extend(d['pos'])
        passages.extend(d['neg'])
        if 0 not in res and len(passages) in res:
            res = [e - 1 for e in res]
        except_res = [i for i in res if i >= len(passages)]
        for e in except_res:
            res.remove(e)
        if len(res) < len(passages):
            print(res)
            for i in range(len(passages)):
                if i not in res:
                    res.append(i)

        d['pos'] = []
        d['neg'] = []
        for i in res[:1]:
            d['pos'].append(passages[i])
        for i in res[1:]:
            d['neg'].append(passages[i])
        d['pos_scores'] = [1]
        d['neg_scores'] = [1 / (i + 1) for i in range(len(res) - 1)]

    return train_data
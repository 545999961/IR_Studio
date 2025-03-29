import argparse
import os
import json
import copy
import random
import time
import shutil

from ir_studio.src.data_generation import generate_query, generate_answer, generate_llm_sft_train_data
from ir_studio.src.generation_agent import GPTAgent, LLMAgent, LLMInstructAgent
from ir_studio.src.prompts import *
from FlagEmbedding import FlagModel
from transformers import AutoTokenizer
from tqdm import trange, tqdm
from generate_retriever_train_data import generate_retriever_train_data
from add_generation_score import get_distill_data


def parse_option():
    parser = argparse.ArgumentParser("")

    parser.add_argument('--query_model_path', type=str, default="gpt-4o-mini")
    parser.add_argument('--query_temperature', type=float, default=0.8)
    parser.add_argument('--query_model_api_key', type=str,
                        default="sk-gcEFxmHmuS2vTQSXA3691eFd78264269B8D1Ce67B749Ac1a")
    parser.add_argument('--query_model_base_url', type=str, default="https://api.key77qiqi.cn/v1")

    parser.add_argument('--generate_model_path', type=str, default="/share/chaofan/models/Meta-Llama-3-8B")
    parser.add_argument('--generate_model_lora_path', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=300)
    parser.add_argument('--model_type', type=str, default="llm")
    parser.add_argument('--train_num', type=int, default=10000)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--retrieval_model_name', type=str, default="/share/chaofan/models/models/bge-large-en-v1.5")
    parser.add_argument('--pooling_method', type=str, default='cls')
    parser.add_argument('--retrieval_query_prompt', type=str,
                        default="Represent this sentence for searching relevant passages: ")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="/share/chaofan/code/ir_studio/result/CovidQA")
    parser.add_argument('--use_answer_retrieval', type=bool, default=False)
    parser.add_argument('--filter_data', type=bool, default=False)
    parser.add_argument('--filter_num', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--emb_save_dir', type=str, default=None)
    parser.add_argument('--ignore_prefix', type=bool, default=False)
    parser.add_argument('--etype', type=str, default='answer')
    parser.add_argument('--linear_path', type=str, default=None)
    parser.add_argument('--normalize_embeddings', type=str, default='True')
    parser.add_argument('--neg_type', type=str, default='random')
    parser.add_argument('--distill_data_path', type=str, default=None)

    opt = parser.parse_args()

    return opt


def main(opt):
    query_model_path = opt.query_model_path
    query_temperature = opt.query_temperature
    query_model_api_key = opt.query_model_api_key
    query_model_base_url = opt.query_model_base_url

    generate_model_path = opt.generate_model_path
    generate_model_lora_path = opt.generate_model_lora_path
    temperature = opt.temperature
    gpu_memory_utilization = opt.gpu_memory_utilization
    tensor_parallel_size = opt.tensor_parallel_size
    top_p = opt.top_p
    max_tokens = opt.max_tokens
    model_type = opt.model_type
    train_num = opt.train_num
    train_ratio = opt.train_ratio
    retrieval_model_name = opt.retrieval_model_name
    pooling_method = opt.pooling_method
    retrieval_query_prompt = opt.retrieval_query_prompt
    max_length = opt.max_length
    batch_size = opt.batch_size
    chunk_size = opt.chunk_size
    dataset_path = opt.dataset_path
    output_dir = opt.output_dir
    filter_data = opt.filter_data
    filter_num = opt.filter_num
    use_answer_retrieval = opt.use_answer_retrieval
    dataset_name = opt.dataset_name
    emb_save_dir = opt.emb_save_dir
    ignore_prefix = opt.ignore_prefix
    etype = opt.etype
    linear_path = opt.linear_path
    normalize_embeddings = opt.normalize_embeddings
    distill_data_path = opt.distill_data_path
    if normalize_embeddings == 'False':
        normalize_embeddings = False
    else:
        normalize_embeddings = True
    neg_type = opt.neg_type

    try:
        os.makedirs('/'.join(queries_output_dir.split('/')[:-1]), exist_ok=True)
    except:
        pass

    """
    dataset_path - data name - corpus.json
    output_dir - data name - queries.jsonl / answers.json / train_retriever.json
    """

    llm_for_query = GPTAgent(
        model_name=query_model_path,
        api_key=query_model_api_key,
        base_url=query_model_base_url,
        temperature=query_temperature
    )

    seed = str(time.time() + random.random())
    prompts_input_dir = os.path.join('tmp_path', seed, 'input')
    prompts_output_dir = os.path.join('tmp_path', seed, 'output')
    os.makedirs(prompts_input_dir, exist_ok=True)
    os.makedirs(prompts_output_dir, exist_ok=True)

    generate_flag = False
    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        os.makedirs(tmp_output_dir, exist_ok=True)
        queries_output_dir = os.path.join(tmp_output_dir, 'queries.json')
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')
        retrieval_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')
        if file_path != 'cqadupstack':
            corpus_path = os.path.join(dataset_path, file_path, 'corpus.json')

            corpus = json.load(open(corpus_path))
        else:
            corpus = []
            for sub_file in os.listdir(os.path.join(dataset_path, file_path)):
                if not os.path.isdir(os.path.join(dataset_path, file_path, sub_file)):
                    continue
                corpus_path = os.path.join(dataset_path, file_path, sub_file, 'corpus.json')

                corpus.extend(json.load(open(corpus_path)))
        old_corpus = copy.deepcopy(corpus)
        random.shuffle(corpus)
        if train_ratio is not None:
            train_num = int(train_ratio * len(corpus))
        corpus = corpus[:train_num]

        if generate_model_lora_path is not None and not os.path.exists(generate_model_lora_path):
            generate_model_lora_path = None

        ### generate queries for each corpus
        if os.path.exists(queries_output_dir):
            queries_corpus = json.load(open(queries_output_dir))
            queries_corpus = queries_corpus[:train_num]
            corpus = [c['passage'] for c in queries_corpus]
        else:
            prompts = [get_query_generation_prompt(file_path, c[:8000]) for c in corpus]
            generated_queries = llm_for_query.generate(prompts)
            qualities_prompts = [get_quality_control_prompt(file_path, q, c) for (q, c) in
                                 zip(generated_queries, corpus)]
            # print(corpus)
            # print(generated_queries)
            generated_qualities = llm_for_query.generate(qualities_prompts)
            print(generated_qualities)
            # sys.exit()
            queries_corpus = []
            for i in range(len(generated_qualities)):
                if '1' in generated_qualities[i]:
                    queries_corpus.append(
                        {
                            'query': generated_queries[i],
                            'passage': corpus[i]
                        }
                    )
            corpus = [e['passage'] for e in queries_corpus]

            with open(queries_output_dir, 'w') as f:
                json.dump(queries_corpus, f)

        if os.path.exists(answers_output_dir):
            pass
        else:
            prompts = [get_additional_info_generation_prompt(file_path, qc['query']) for qc in queries_corpus]
            # prompts = [get_additional_info_generation_train_prompt(file_path, qc['query'], qc['passage']) for qc in queries_corpus]

            with open(os.path.join(prompts_input_dir, f'{file_path}.json'), 'w') as f:
                json.dump(prompts, f)
            generate_flag = True

    ### run bash to call LLM
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
                --num_gpus 8 \
                --rm_tmp True "
        os.system(cmd)

    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        os.makedirs(tmp_output_dir, exist_ok=True)
        queries_output_dir = os.path.join(tmp_output_dir, 'queries.json')
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')
        retrieval_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')

        if generate_model_lora_path is not None and not os.path.exists(generate_model_lora_path):
            generate_model_lora_path = None

        ### generate queries for each corpus
        queries_corpus = json.load(open(queries_output_dir))
        queries_corpus = queries_corpus[:train_num]
        corpus = [c['passage'] for c in queries_corpus]

        if os.path.exists(answers_output_dir):
            pass
        else:
            generate_answers = json.load(open(os.path.join(prompts_output_dir, f'{file_path}.json')))

            for i in range(len(generate_answers)):
                queries_corpus[i]['answer'] = generate_answers[i]

            with open(answers_output_dir, 'w') as f:
                json.dump(queries_corpus, f)

    if os.path.exists(os.path.join('tmp_path', seed)):
        shutil.rmtree(os.path.join('tmp_path', seed))

    retrieval_model = FlagModel(retrieval_model_name,
                                query_instruction_for_retrieval=retrieval_query_prompt,
                                pooling_method=pooling_method,
                                use_fp16=True,
                                linear_path=linear_path,
                                normalize_embeddings=normalize_embeddings)
    f_all = open(os.path.join(output_dir, 'train.jsonl'), 'w')
    for file_path in os.listdir(dataset_path):
        if dataset_name is not None:
            if file_path != dataset_name:
                continue
        if not os.path.isdir(os.path.join(dataset_path, file_path)):
            continue
        tmp_output_dir = os.path.join(output_dir, file_path)
        answers_output_dir = os.path.join(tmp_output_dir, 'answers.json')
        retrieval_data_output_dir = os.path.join(tmp_output_dir, 'train.jsonl')
        print(file_path, time.time())
        if file_path != 'cqadupstack':
            corpus_path = os.path.join(dataset_path, file_path, 'corpus.json')

            corpus = json.load(open(corpus_path))
        else:
            corpus = []
            for sub_file in os.listdir(os.path.join(dataset_path, file_path)):
                if not os.path.isdir(os.path.join(dataset_path, file_path, sub_file)):
                    continue
                corpus_path = os.path.join(dataset_path, file_path, sub_file, 'corpus.json')

                corpus.extend(json.load(open(corpus_path)))
        print(time.time())
        old_corpus = corpus

        queries_corpus = json.load(open(answers_output_dir))
        queries_corpus = queries_corpus[: train_num]
        ### generate bge train data {'query', 'pos', 'neg'}
        if use_answer_retrieval:
            for i in range(len(queries_corpus)):
                instruction = TASK_DICT[file_path]
                query = queries_corpus[i]['query']
                answer = 'Generate the topic about this passage: ' + queries_corpus[i]['answer']
                # answer = query + '\n' + queries_corpus[i]['answer']
                # answer = queries_corpus[i]['answer']
                queries_corpus[i]['query'] = query
                queries_corpus[i]['answer'] = answer
                # queries_corpus[i]['query'] = "Instruction: {instruction}\nQuery: {query}\nPossible response: {answer}".format(instruction=instruction, query=query, answer=answer)
        corpus = [c['passage'] for c in queries_corpus]

        if train_ratio is not None:
            train_num = int(train_ratio * len(corpus))
        # for c in old_corpus[:train_num * 30]:
        # for c in old_corpus:
        #     if c not in corpus:
        #         corpus.append(c)
        corpus.extend(old_corpus)

        print('corpus length:', len(corpus), ';', 'queries length:', len(queries_corpus))

        if emb_save_dir is not None:
            if file_path in ['cqadupstack', 'webis-touche2020']:
                emb_save_path = os.path.join(emb_save_dir, file_path, 'tmp_corpus.npy')
            else:
                emb_save_path = os.path.join(emb_save_dir, file_path, 'corpus.npy')
        else:
            emb_save_path = None

        # if not os.path.exists(retrieval_data_output_dir):
        #     bge_train_data = generate_retriever_train_data(retrieval_model, batch_size, max_length,
        #                                             queries_corpus, 'passage', corpus, filter_data, filter_num,
        #                                             emb_save_path, ignore_prefix, etype, neg_type)
        # else:
        #     bge_train_data = []
        #     for line in open(retrieval_data_output_dir):
        #         bge_train_data.append(json.loads(line))
        bge_train_data = generate_retriever_train_data(retrieval_model, batch_size, max_length,
                                                 queries_corpus, 'passage', corpus, filter_data, filter_num,
                                                 emb_save_path, ignore_prefix, etype, neg_type)

        del retrieval_model

        if distill_data_path is None:
            tmp_tokenizer = AutoTokenizer.from_pretrained('/share/chaofan/models/Meta-Llama-3-8B-Instruct')
            prompts = []
            for d in tqdm(bge_train_data, desc='generate train data'):
                passages = []
                passages.extend(d['pos'])
                passages.extend(d['neg'])
                passages_ids = tmp_tokenizer(passages, max_length=512, truncation=True)['input_ids']
                passages = tmp_tokenizer.batch_decode(passages_ids)
                prompts.append(
                    rank_prompt.format(
                        num=len(passages),
                        query=d['query'],
                        passages='\n'.join([f'[{i}] {passages[i]}' for i in range(len(passages))])
                    )
                )

            bge_train_data = get_distill_data(
                model_name_or_path=generate_model_path,
                model_type=model_type,
                train_data=bge_train_data,
                prompts=prompts,
                batch_size=16,
            )
        else:
            new_bge_train_data = []
            raw_path = os.path.join(
                distill_data_path,
                retrieval_data_output_dir.split('train_data/single/result/BEIR/')[-1]
            )
            with open(raw_path) as f:
                for line in f:
                    new_bge_train_data.append(json.loads(line))
            for i in range(len(bge_train_data)):
                new_bge_train_data[i]['answer'] = bge_train_data[i]['answer']
                new_bge_train_data[i]['neg_answer'] = bge_train_data[i]['neg_answer']
            bge_train_data = new_bge_train_data

        with open(retrieval_data_output_dir, 'w') as f:
            for d in bge_train_data:
                f.write(json.dumps(d) + '\n')
                f_all.write(json.dumps(d) + '\n')
    f_all.close()


if __name__ == "__main__":
    opt = parse_option()
    main(opt)
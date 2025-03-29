import json
import random
import os

import numpy as np
from FlagEmbedding import FlagModel
from tqdm import tqdm, trange
from typing import List, Union

from ir_studio.src.evaluation.evaluate import search

def generate_retriever_train_data(
    retrieval_model,
    batch_size: int = 512,
    max_length: int = 512,
    queries_corpus: Union[List[dict], List[List[dict]]] = None,
    dtype: str = 'passage',
    corpus: List[str] = None,
    filter_data: bool = False,
    filter_num: int = 20,
    emb_save_path: str = None,
    ignore_prefix: bool = False,
    etype: str = 'answer',
    neg_type: str = 'hard'
):
    if not isinstance(queries_corpus[0], list):
        if corpus is None:
            corpus = [d[dtype] for d in queries_corpus]
        queries = [d['query'] for d in queries_corpus]
        answers = [d['answer'] for d in queries_corpus]

        queries_emb = retrieval_model.encode_queries(queries, batch_size=batch_size, max_length=max_length) # * 0.8 + retrieval_model.encode_corpus(answers, batch_size=batch_size, max_length=max_length, etype=etype) * 0.2
        # queries_emb = retrieval_model.encode_queries(queries, batch_size=batch_size, max_length=max_length)
        answers_emb = retrieval_model.encode_corpus(answers, batch_size=batch_size, max_length=max_length, etype=etype)
        if emb_save_path is not None:
            if os.path.exists(emb_save_path):
                if ignore_prefix:
                    doc_emb = np.vstack(
                            (
                                retrieval_model.encode_corpus(corpus[: len(queries_emb)], batch_size=batch_size, max_length=max_length),
                                np.load(emb_save_path)
                            )
                    )
                else:
                    doc_emb = np.load(emb_save_path)
            else:
                doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
                try:
                    os.makedirs('/'.join(emb_save_path.split('/')[:-1]), exist_ok=True)
                except:
                    pass
                if ignore_prefix:
                    np.save(emb_save_path, doc_emb[len(queries_emb): ])
                else:
                    np.save(emb_save_path, doc_emb)
        else:
            doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
        
        print('len doc emb:', len(doc_emb))

        all_scores, all_indices = search(queries_emb, doc_emb, 2000)
        _, all_answers_indices = search(answers_emb, doc_emb, 2000)

        train_data = []

        find_idxs = []
        for i in range(len(all_indices)):
            if i in list(all_indices[i]):
                find_idxs.append(list(all_indices[i]).index(i))
            else:
                find_idxs.append(-1)
        print(find_idxs)

        answers_find_idxs = []
        for i in range(len(all_answers_indices)):
            if i in list(all_answers_indices[i]):
                answers_find_idxs.append(list(all_answers_indices[i]).index(i))
            else:
                answers_find_idxs.append(-1)

        for i in trange(len(queries), desc='generate train set'):

            if find_idxs[i] == -1: # remove false pairs
                # continue
                # neg_ids = random.sample(list(range(len(corpus))), k=50)
                neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
            else:            
                uses_idx = -1
                for j in range(find_idxs[i] + 1, 2000):
                    if all_scores[i][j] <= all_scores[i][find_idxs[i]] * 0.95:
                        uses_idx = j
                        break
                if uses_idx == -1:
                    # continue
                    # neg_ids = random.sample(list(range(len(corpus))), k=50)
                    neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
                else:
                    neg_ids = list(all_indices[i][uses_idx: uses_idx + 50])
            # neg_ids = list(all_indices[i][:50])
            if neg_type == 'random':
                neg_ids = random.sample(list(range(len(corpus))), k=50)
            elif neg_type == 'hard':
                # neg_ids = list(all_indices[i][:50])
                neg_ids = random.sample(list(all_indices[i][30:200]), k=50)
                tmp_ids = [(e, list(all_indices[i]).index(e)) for e in neg_ids]
                tmp_ids = sorted(tmp_ids, key=lambda x: x[1])
                neg_ids = [e[0] for e in tmp_ids]
            else:
                tmp_ids = [(e, list(all_indices[i]).index(e)) for e in neg_ids]
                tmp_ids = sorted(tmp_ids, key=lambda x: x[1])
                neg_ids = [e[0] for e in tmp_ids]
            

            if answers_find_idxs[i] == -1: # remove false pairs
                # continue
                neg_answers_ids = random.sample(list(range(len(corpus))), k=50)
            else:            
                uses_idx = -1
                for j in range(answers_find_idxs[i] + 1, 2000):
                    if all_scores[i][j] <= all_scores[i][answers_find_idxs[i]] * 0.95:
                        uses_idx = j
                        break
                if uses_idx == -1:
                    # continue
                    neg_answers_ids = random.sample(list(range(len(corpus))), k=50)
                else:
                    neg_answers_ids = list(all_answers_indices[i][uses_idx: uses_idx + 50])

            query = queries[i]
            answer = answers[i]
            pos = [corpus[i]]
            negs = [corpus[j] for j in neg_ids]
            while pos[0] in negs:
                negs.remove(pos[0])
            new_negs = []
            for e in negs:
                if e not in new_negs and len(new_negs) < 15:
                    new_negs.append(e)
            negs = new_negs

            # neg_answers_ids = random.sample(list(range(len(corpus))), k=50)
            # neg_answers_ids = all_answers_indices[i][:50]
            negs_answer = [corpus[j] for j in neg_answers_ids]
            while pos[0] in negs_answer:
                negs_answer.remove(pos[0])
            new_negs_answer = []
            for e in negs_answer:
                if e not in new_negs_answer and len(new_negs_answer) < 15:
                    new_negs_answer.append(e)
            negs_answer = new_negs_answer

            train_data.append(
                {
                    'query': query,
                    'answer': answer,
                    'pos': pos,
                    'neg': negs,
                    'neg_answer': negs_answer
                }
            )
            # print(len(train_data))
        
        if filter_data:
            print(filter_data)
            new_train_data = []
            for i in range(len(all_indices)):
                if i in list(all_indices[i]):
                    seached_idx = list(all_indices[i]).index(i)
                else:
                    seached_idx = len(all_indices) + 999
                if seached_idx < filter_num:
                    new_train_data.append(train_data[i])
            train_data = new_train_data
    else:
        print('error')
        # queries_corpus_list = queries_corpus
        # queries_list = []
        # corpus_list = []
        # answers_list = []
        # for qc in queries_corpus_list:
        #     queries_list.append([d['query'] for d in qc])
        #     corpus_list.append([d[dtype] for d in qc])
        #     answers_list.append([d['answer'] for d in qc])
        
        # all_indices_list = []
        # for queries, corpus in zip(queries_list, corpus_list):
        #     queries_emb = retrieval_model.encode_queries(queries, batch_size=batch_size, max_length=max_length)
        #     doc_emb = retrieval_model.encode_corpus(corpus, batch_size=batch_size, max_length=max_length)
        #     all_scores, all_indices = search(queries_emb, doc_emb)
        #     all_indices_list.append(list([list(l) for l in all_indices]))

        # pos_indexs = []
        # for i in range(len(all_indices_list[0])):
        #     pos_indexs.append([])
        #     for all_indices in all_indices_list:
        #         if i in list(all_indices[i]):
        #             seached_idx = list(all_indices[i]).index(i)
        #         else:
        #             seached_idx = len(all_indices) + 999
        #         pos_indexs[-1].append(seached_idx)
        
        # train_data = []
        # for i in range(len(all_indices_list[0])):
        #     min_idx = pos_indexs[i].index(min(pos_indexs[i]))
        #     # if min_idx == 0:
        #     #     if random.random() >= 0.5:
        #     #         continue
        #     if filter_data:
        #         if min(pos_indexs[i]) < filter_num:
        #             selected_indices = list(all_indices_list[min_idx][i])
        #             if i in selected_indices:
        #                 selected_indices.remove(i)
        #             neg_ids = random.choices(selected_indices[20:], k=30)
        #             pos = [corpus[i]]
        #             negs = [corpus[j] for j in neg_ids]
        #             while pos[0] in negs:
        #                 negs.remove(pos[0])
        #             train_data.append(
        #                 {
        #                     'query': queries_list[min_idx][i],
        #                     'answer': answers_list[min_idx][i],
        #                     'pos': pos,
        #                     'neg': list(set(negs))
        #                 }
        #             )
        #     else:
        #         selected_indices = list(all_indices_list[min_idx][i])
        #         if i in selected_indices:
        #             selected_indices.remove(i)
        #         neg_ids = random.choices(selected_indices[20:], k=25)
        #         pos = [corpus[i]]
        #         negs = [corpus[j] for j in neg_ids]
        #         while pos[0] in negs:
        #             negs.remove(pos[0])
        #         train_data.append(
        #             {
        #                 'query': queries_list[min_idx][i],
        #                 'answer': answers_list[min_idx][i],
        #                 'pos': pos,
        #                 'neg': list(set(negs))
        #             }
        #         )

    print(len(train_data))

    return train_data
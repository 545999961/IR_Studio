import time
import json

from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from functools import partial


class GPTAgent():
    def __init__(self, model_name: str = "gpt-4o-mini", base_url: str = None, api_key: str = None,
                 temperature: float = 0, top_p: float = 1, max_tokens: int = 300, n: int = 1):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.n = n

    def generate_single(self, prompt, use_beam_search: bool = False):
        llm = OpenAI(base_url=self.base_url, api_key=self.api_key)
        while True:
            try:
                if use_beam_search:
                    completion = llm.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        n=self.n,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        extra_body={
                            'use_beam_search': True,
                            'best_of': 10
                        }
                    )
                else:
                    completion = llm.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        n=self.n,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )
                if len(completion.choices) == 0:
                    return completion.choices[0].message.content.strip('\n').strip()
                return [e.message.content.strip('\n').strip() for e in completion.choices]
            except Exception as e:
                print(str(e))
                if 'Request timed out' in str(e):
                    return ''
                time.sleep(5)

    def generate(self, prompts, use_beam_search: bool = False, thread_count: int = None):
        if isinstance(prompts, str):
            prompts = [prompts]

        if thread_count is None:
            thread_count = cpu_count()

        generate_with_beam = partial(self.generate_single, use_beam_search=use_beam_search)

        results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Map the fixed function to the prompts
            results = list(tqdm(executor.map(generate_with_beam, prompts), total=len(prompts)))

        return results

    def generate_single_direct(self, prompt):
        llm = OpenAI(base_url=self.base_url, api_key=self.api_key)
        while True:
            try:
                completion = llm.chat.completions.create(
                    model=self.model_name,
                    messages=json.loads(prompt),
                    n=1,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
                return completion.choices[0].message.content.strip('\n').strip()
            except Exception as e:
                print(e)
                time.sleep(5)

    def generate_direct(self, prompts, thread_count: int = None):
        if isinstance(prompts, str):
            prompts = [prompts]

        if thread_count is None:
            thread_count = cpu_count()

        prompts = [json.dumps(p) for p in prompts]

        results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(self.generate_single_direct, prompts), total=len(prompts)))

        return results
import time

from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

class GPTAgent():
    def __init__(self, model_name: str = "gpt-4o-mini", base_url: str = None, api_key: str = None, temperature: float = 0):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature

    def generate_single(self, prompt):
        llm = OpenAI(base_url=self.base_url, api_key=self.api_key)
        while True:
            try:
                completion = llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    n=1,
                    temperature=self.temperature
                )
                return completion.choices[0].message.content.strip('\n').strip()
            except Exception as e:
                print(e)
                time.sleep(5)
    
    def generate(self, prompts, thread_count: int = None):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if thread_count is None:
            thread_count = cpu_count()

        results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(tqdm(executor.map(self.generate_single, prompts), total=len(prompts)))
        
        return results
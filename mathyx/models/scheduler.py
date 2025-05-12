import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

"""
SCHEDULER.PY
============================================================
An internal tool to parallelize API requests and utilize
retries when a request fails

"""

class Scheduler:
    def __init__(self, function, concurrent_requests=8, max_retries=8, delay=0.1):
        self.function = function
        self.concurrent_requests = concurrent_requests
        self.max_retries = max_retries
        self.delay = delay
    
    def run(self, prompts, **kwargs):
        total = len(prompts)
        completed = 0
        output = [[]] * total

        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            future_to_index = {
                executor.submit(self.run_with_retries, prompt, **kwargs): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                output[idx] = future.result()
                completed += 1
                if completed == total:
                    return output

    def run_with_retries(self, prompt, **kwargs):
        try_cnt = 0
        while try_cnt < self.max_retries:
            try:
                output = self.function(prompt, **kwargs)
                time.sleep(self.delay)
                return output
            except Exception as e:
                time.sleep(self.delay)
                # if api error is not due to rate limit, try again
                if "rate limit" not in str(e).lower() and "429" not in str(e):
                    try_cnt += 1
        return ""

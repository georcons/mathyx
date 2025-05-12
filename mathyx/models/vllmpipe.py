import os
import json
import random
import string
from typing import List
from ..abstract import Pipeline
from ..types import PromptResult, Evaluation

from vllm import LLM, SamplingParams

"""
VLLMPIPE.PY
====================================================
A pipeline to run vLLM.

"""

class VllmPipe(Pipeline):
    def __init__(
        self,
        model : str | None = 'tiiuae/falcon-rw-1b',
        *,
        temperature : float | None = 1.0,
        max_tokens : int | None = None,
        gpu_count : int | None = 1
    ):
        self.model = model
        self.config = {}
        self.config['temperature'] = temperature
        self.config['max_tokens'] = max_tokens
        self.config['gpu_count'] = gpu_count

    def send(
        self,
        query : str,
        *,
        response_count : int | None = 1
    ) -> PromptResult:
        raise NotImplementedError

    def sync(
        self,
        query : List[str],
        *,
        response_count : int | None = 1
    ) -> Evaluation:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(self.config['gpu_count'])])
        # Load LLM
        llm = LLM(model=self.model, gpu_memory_utilization=0.9, tensor_parallel_size=self.config['gpu_count'])
        # Set up sampling parameters
        sampling_params = (SamplingParams(temperature=self.config['temperature'], max_tokens=self.config['max_tokens']) if self.config['max_tokens'] is not None 
                            else SamplingParams(temperature=self.config['temperature']))
        # Prepare prompts
        prompts = [prompt for prompt in query for _ in range(response_count)]
        # Run LLM
        generated = llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text.strip() for output in generated]
        grouped_outputs = self.__split_array(outputs, int(len(outputs) / response_count))
        return Evaluation.from_list(grouped_outputs)

    def async_send(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1,
        batch_directory : str | None = "./"
    ) -> str:
        raise NotImplementedError

    def retrieve(
        self,
        batch_id : str
    ) -> Evaluation | None:
        raise NotImplementedError

    def use_multiple_sync(self): return True

    def __split_array(self, arr, n):
        avg = len(arr) // n
        remainder = len(arr) % n
        result = []
        start = 0

        for i in range(n):
            end = start + avg + (1 if i < remainder else 0)
            result.append(arr[start:end])
            start = end

        return result

import os
import json
import random
import openai
import string
from typing import List
from ..abstract import Pipeline
from ..types import PromptResult, Evaluation

"""
GPTPIPE.PY
====================================================
A pipeline to communicate with OpenAI's API

"""

class OpenAIPipe(Pipeline):
    def __init__(
        self,
        model : str | None = 'gpt-4o-mini',
        **kwargs
    ):
        self.model = model
        self.config = kwargs

    def send(
        self,
        query : str,
        *,
        response_count : int | None = 1
    ) -> PromptResult:
        message = [{"role": "user", "content": query}]
        config = self.config
        results = [openai.chat.completions.create(
            model=self.model,
            messages=message,
            **config
        ).choices[0].message.content for _ in range(response_count)]
        return PromptResult(results)

    def sync(
        self,
        query : List[str],
        *,
        response_count : int | None = 1
    ) -> None:
        raise NotImplementedError

    def async_send(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1,
        batch_directory : str | None = "./"
    ) -> str:
        total_count = len(queries)
        tasks = []
        for index, query in enumerate(queries):
            tasks += self.__generate_query(
                query,
                index,
                total_count,
                response_count=response_count
            )
        
        filename = batch_directory + "openai-" + (''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(24))) + ".jsonl"
        
        with open(filename, 'w') as file:
            for t in tasks:
                file.write(json.dumps(t) + '\n')
        
        batch_file = openai.files.create(
            file=open(filename, 'rb'),
            purpose='batch'
        )

        batch_job = openai.batches.create(
            input_file_id=batch_file.id,
            endpoint='/v1/chat/completions',
            completion_window='24h'
        )

        return batch_job.id

    def retrieve(
        self,
        batch_id : str
    ) -> Evaluation | None:
        batch = openai.batches.retrieve(batch_id)

        if batch.status != 'completed': return None

        output_file_id = batch.output_file_id
        output_file = openai.files.content(output_file_id).text
        output_lines = output_lines.splitlines()

        output = []

        id_init = json.loads(output_lines[0].strip())['custom_id']
        blocks_init = id_init.split("-")

        if len(blocks_init) != 4:
            return None

        total_count = int(blocks_init[3])

        for i in range(total_count):
            output.append([])
        
        for line in output_lines:
            obj = json.loads(line.strip())
            blocks = obj['custom_id'].split("-")

            if len(blocks) == 4:
                current_index = int(blocks[2])
                result = obj['response']['body']['choices'][0]['message']['content']
                output[current_index].append(result)

        return Evaluation.from_list(output)

    def use_multiple_sync(self): return False

    def __generate_query(
        self,
        query : str, 
        index : int,
        total_count : int,
        *,
        response_count : int | None = 1
    ) -> str:
        body = self.config
        body['model'] = self.model
        body['messages'] = [{'role': 'user', 'content': query}]
        obj = [{
            'custom_id': f'{i}-{response_count}-{index}-{total_count}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': body
        } for i in range(response_count)]
        return obj

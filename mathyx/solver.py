from typing import List
from .abstract import Pipeline, Formatizer
from .models import Pipelines
from .types import Model, Pass, PromptResult, Evaluation
from .utils import DefaultFormatizer

class Solver:
    __pipeline : Pipeline
    formatizer : Formatizer

    def __init__(
        self,
        *,
        pipeline : Pipeline | Model | None = None,
        model : str | None = None,
        load : bool | None = True,
        **kwargs
    ):
        if isinstance(pipeline, Model):
            pipeline = Pipelines.from_model(pipeline)
        _pipe = pipeline if pipeline is not None else Pipelines.OpenAI()
        self.__pipeline = Pipelines.wrap(_pipe)
        if model is not None:
            self.__pipeline.model = model
        self.formatizer = DefaultFormatizer()
        if load: self.__pipeline.load()
        for key in kwargs:
            self.__pipeline.config[key] = kwargs[key]

    @property
    def pipeline(self):
        return self.__pipeline
    
    @pipeline.setter
    def pipeline(self, _pipe):
        self.__pipeline = Pipelines.wrap(_pipe)
        self.__pipeline.load()

    @property
    def template(self):
        return self.formatizer.template

    @template.setter
    def template(self, value):
        self.formatizer.template = value

    def solve(
        self,
        problems : List[str],
        *,
        attempts : int | None = 1,
        **kwargs
    ) -> Evaluation:
        queries = self.formatizer.compile_problems(problems, **kwargs)
        results = self.__pipeline.sync(queries, response_count=attempts)
        results.extract_answer = self.formatizer.extract_answer
        results.set_prompts(queries)
        results.set_statements(problems)
        return results

    def async_solve(
        self,
        problems : List[str],
        *,
        attempts : int | None = 1,
        **kwargs
    ) -> str:
        queries = self.formatizer.compile_problems(problems, **kwargs)
        return self.__pipeline.async_send(queries, response_count=attempts)
    
    def retrieve(
        self,
        batch_id : str,
        *,
        statements : List[str] | None = None,
        queries : List[str] | None = None
    ) -> Evaluation:
        results = self.__pipeline.retrieve(batch_id)
        results.extract_answer = self.formatizer.extract_answer

        if queries is not None:
            results.set_prompts(queries)
        
        if statements is not None:
            results.set_statements(statements)

        return results

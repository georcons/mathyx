from typing import List
from ..abstract import Pipeline
from .scheduler import Scheduler
from ..types import PromptResult, Evaluation

"""
MODELWRAPPER.PY
===========================================================
A wrapper that adds additional functionatily (concurency
and retries) to a already created pipeline

"""

class _WrappedPipeline(Pipeline):
    __wrapped : Pipeline
    __use_scheduler : bool
    __scheduler : Scheduler

    def __init__(
        self,
        _pipeline : Pipeline
    ):
        self.__wrapped = _pipeline
        self.__use_scheduler = not _pipeline.use_multiple_sync()
        if self.__use_scheduler:
            self.__scheduler = Scheduler(self.__wrapped.send)

    def load(self) -> None:
        return self.__wrapped.load()

    def send(
        self,
        query : str,
        *,
        response_count : int | None = 1
    ) -> PromptResult | None:
        return self.__wrapped.send(query, response_count=response_count)

    def sync(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1
    ) -> Evaluation | None:
        if self.__use_scheduler:
            responses = self.__scheduler.run(queries, response_count=response_count)
            return Evaluation(responses)
        else:
            return self.__wrapped.sync(queries, response_count=response_count)
    
    def async_send(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1
    ) -> str:
        return self.__wrapped.async_send(queries, response_count=response_count)

    def retrieve(
        self,
        batch_id : str
    ) -> Evaluation | None:
        return self.__wrapped.retrieve(batch_id)

    def use_multiple_sync(self): return True

    @property
    def concurrent_requests(self):
        return self.__scheduler.concurrent_requests
    
    @concurrent_requests.setter
    def concurrent_requests(
        self, 
        value : int
    ):
        if value < 1:
            raise Exception("concurrent_requests must be a possitive integer")
        else: self.__scheduler.concurrent_requests = value

    @property
    def max_retries(self):
        return self.__scheduler.max_retries

    @max_retries.setter
    def max_retries(
        self,
        value : int
    ):
        if value < 1:
            raise Exception("max_retries must be a possitive integer")
        else: self.__scheduler.max_retries = max_retries

    @property
    def config(self):
        return self.__wrapped.config

    @property
    def model(self):
        return self.__wrapped.model

    @model.setter
    def model(self, value : str):
        self.__wrapped.model = value

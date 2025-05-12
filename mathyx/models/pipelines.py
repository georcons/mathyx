from .modelwrapper import _WrappedPipeline
from .gptpipe import OpenAIPipe
from .pipeline_sources import PIPELINE_SOURCES
from ..abstract import Pipeline
from ..types import Model

class Pipelines:
    @staticmethod
    def wrap(
        _pipeline : Pipeline
    ) -> Pipeline:
        return _WrappedPipeline(_pipeline)

    @staticmethod
    def OpenAI(
        model : str | None = 'gpt-4o-mini',
        **kwargs
    ) -> Pipeline:
        return OpenAIPipe(model, **kwargs)

    @staticmethod
    def from_model(
        model : Model
    ) -> Pipeline:
        if model.source not in PIPELINE_SOURCES:
            raise Exception(f'Source {model.source} not found in pipeline_sources.py')
        pipe_ = PIPELINE_SOURCES[model.source]
        return pipe_(model.model, model.config)

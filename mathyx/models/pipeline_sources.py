from .gptpipe import OpenAIPipe
from .vllmpipe import VllmPipe

PIPELINE_SOURCES = {
    'OpenAI' : OpenAIPipe,
    'vLLM' : VllmPipe
}
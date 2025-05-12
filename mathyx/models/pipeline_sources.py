from .gptpipe import OpenAIPipe
from .VllmPipe import VllmPipe

PIPELINE_SOURCES = {
    'OpenAI' : OpenAIPipe,
    'vLLM' : VllmPipe
}
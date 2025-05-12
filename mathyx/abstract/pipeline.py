from abc import ABC, abstractmethod
from typing import List

class Pipeline(ABC):
    # Initialize with internal model name (for different models inside the pipeline)
    # and **kwargs with custom model parameters
    def __init__(
        model : str | None = None,
        **kwargs
    ):
        self.model = model
        self.config = kwargs

    # Load resources if needed
    def load(self) -> None:
        return None

    # Retrieve synchronically model response from single query
    @abstractmethod
    def send(
        self,
        query : str,
        *,
        response_count : int | None = 1 # Responses to generate
    ):
        pass

    # Retrieve synchronically multiple model responses from multiple queries
    @abstractmethod
    def sync(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1 # Responses to generate per query
    ):
        pass

    # Send batch of multiple queries to the model to be computed asynchronically
    @abstractmethod
    def async_send(
        self,
        queries : List[str],
        *,
        response_count : int | None = 1 # How many responses to generate per query
    ) -> str: # Return a Batch ID to use later for retrieval
        pass

    # Retrieve responses from a batch sent via send_batch() using the returned ID
    @staticmethod
    @abstractmethod
    def retrieve(
        self,
        batch_id : str
    ): # Return a list of the results
        pass

    # Indicate if you've implemented the .sync method to handle
    # multiple queries synchronically
    @staticmethod
    def use_multiple_sync(self) -> bool:
        return False
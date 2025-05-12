from abc import ABC, abstractmethod
from typing import List

class Formatizer(ABC):
    @abstractmethod
    def compile_problems(
        self, 
        statements : List[str], 
        **kwargs
    ) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def extract_answer(
        solution : str
    ) -> str:
        pass
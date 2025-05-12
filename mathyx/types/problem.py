from typing import List
from .result import Pass

class FalconProblem:
    _id : int = -1
    statement : str
    answer : str
    solution : str
    generations : List[Pass] | None = None
    info : dict

    def __getitem__(
        self, 
        key : str
    ) -> object:
        if not isinstance(key, str):
            raise TypeError('Key must be a string')
        if key == '_id': return self._id
        elif key == 'statement': return self.statement
        elif key == 'answer': return self.answer
        elif key == 'solution': return self.solution
        else: return self.info[key]

    def __setitem__(
        self,
        key : str,
        value : object
    ) -> None:
        if not isinstance(key, str):
            raise TypeError('Key must be a string')
        elif key == '_id': raise Exception('Cannot modify _id key.')
        elif key == 'statement':
            if not isinstance(value, str): raise TypeError('statement must be a string')
            else: self.statement = value
        elif key == 'answer':
            self.answer = str(value)
        elif key == 'solution':
            if not isinstance(value, str): raise TypeError('solution must be a string')
            else: self.solution = value
        else: self.info[key] = value

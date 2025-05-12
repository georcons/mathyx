from datetime import datetime
from typing import List, Callable
from .model import Model
from ..utils import DefaultFormatizer

"""
RESULTS.PY
========================================================
Classes used to encapsulate responses from LLMs.

"""

# Encapsulates the direct result of a prompt
class Pass:
    statement : str | None = None
    user : str | None = None
    assistant : str = ""
    model : Model | None = None
    date : datetime | None = None

    def __init__(
        self,
        A : str,
        B : str | None = None
    ):
        if B is None: self.assistant = A
        else:
            self.user = A
            self.assistant = B

    def __str__(self) -> str:
        return self.assistant

# Encapsulates several passes of the same query
class PromptResult:
    extract_answer : Callable[[str], str] | None
    user : str | None = None
    statement : str | None = None
    correct_answer : str | None = None
    __passes : List[Pass]
    __pass_count : int

    # Initialized from a list of responses corresponding a single query
    # passes through the LLM several times
    def __init__(
        self,
        responses : List[str],
        *,
        prompt : str | None = None
    ):
        self.__pass_count = len(responses)
        self.user = prompt
        self.__passes = [Pass(prompt, response) for response in responses]
        self.extract_answer = None

    @property
    def generations(self):
        return self.__passes

    # Set query for all responses
    def set_prompt(
        self,
        prompt : str
    ) -> None:
        self.prompt = prompt
        for response in self.__passes:
            response.user = prompt

    # Set statement for all responses
    def set_statement(
        self,
        statement : str | None
    ) -> None: 
        self.statement = statement
        for response in self.__passes:
            response.statement = statement

    # Returns the original list
    def to_list(self):
        return [resp.assistant for resp in self.__passes]

    # Returns the number of passes
    def count(self) -> int:
        return self.__pass_count

    # Get a list of answers from the solutions
    def answers(
        self,
        extract_answer : Callable[[str], str] | None = None
    ) -> List[str]:
        extract_answer = (
                            extract_answer 
                            if extract_answer is not None 
                            else self.extract_answer 
                            if self.extract_answer is not None 
                            else DefaultFormatizer.extract_answer
        )
        return [extract_answer(res.assistant) for res in self.__passes]

    # Grade what portion of the answers are correct given a ground-truth
    def grade(
        self,
        ground_truth_answer : str | None = None, 
        /,
        extract_answer : Callable[[str], str] | None = None
    ) -> float:
        if ground_truth_answer is None:
            if self.correct_answer is not None:
                ground_truth_answer = self.correct_answer
            else:
                raise Exception('To grade, either parse answer as an argument, or make sure it is set in the object')
        answers = self.answers(extract_answer)
        return answers.count(str(ground_truth_answer)) / len(answers)

    # System methods
    def __len__(self) -> int:
        return self.count()

    def __iter__(self):
        return iter(self.__passes)

    def __getitem__(self, index):
        return self.__passes[index]

    def __str__(self):
        return str(self.to_list())

# Used to store a list of instances of Prompt
class Evaluation:
    __prompts = List[PromptResult]
    extract_answer : Callable[[str], str] | None

    # Initialize from a list of Prompt
    def __init__(
        self,
        results : List[PromptResult]
    ):
        self.__results = results
        self.extract_answer = None

    # Alternatively, initialize from a list of list of strings
    @staticmethod
    def from_list(
        results : List[List[str]],
        prompts : List[str] | None = None
    ):
        output = []
        if prompts is not None:
            if len(queries) != len(results):
                raise Exception("""When parsing both results and queries, 
                                        number of result lists must correspond to the number of queries.""")
                output = [
                    PromptResult(result, prompt=prompt) for prompt, result in zip(prompt, results)
                ]
        else: output = [PromptResult(result) for result in results]
        return Result(output)

    # Restore the original list of lists
    def to_list(self):
        return [
                prompt_result.to_list() for prompt_result in self.__prompts
        ]

    # Set queries for each result
    def set_prompts(
        self,
        queries : List[str]
    ) -> None:
        if len(queries) != len(self):
            raise Exception('Number of queries must be the same as number of resutls.')
        for query, result in zip(queries, self.__results):
            result.set_prompt(query)

    # Set statements for each result
    def set_statements(
        self,
        statements : List[str]
    ) -> None:
        if len(statements) != len(self):
            raise Exception('Number of statements must be the same as number of resutls.')
        for statement, result in zip(statements, self.__results):
            result.set_statement(statement)

    # Get correct answers from object
    def correct_answers(
        self
    ) -> List[str]:
        return [prompt.correct_answer for prompt in self.__results]

    # Get answers from the results
    def answers(
        self,
        extract_answer : Callable[[str], str] | None = None
    ):
        extract_answer = (
                            extract_answer 
                            if extract_answer is not None 
                            else self.extract_answer 
                            if self.extract_answer is not None 
                            else DefaultFormatizer.extract_answer
        )
        return [result.answers(extract_answer) for result in self.__results]

    # Grade each of the results in the list
    def grade(
        self,
        ground_truth_answers : List[str] | None = None,
        /,
        extract_answer : Callable[[str], str] | None = None
    ) -> List[float]:
        if ground_truth_answers is not None:
            if len(ground_truth_answers) != len(self.__results):
                raise Exception('Number of ground-truth answers must the same at results in the list.')
            return [result.grade(ans, extract_answer=extract_answer) for ans, result in zip(ground_truth_answers, self.__results)]
        else: return [result.grade(extract_answer=extract_answer) for result in self.__results]

    # System methods
    def __len__(self):
        return len(self.__results)

    def __getitem__(self, index):
        return self.__results[index]

    def __iter__(self):
        return iter(self.__results)

    def __str__(self):
        return str(self.to_list())

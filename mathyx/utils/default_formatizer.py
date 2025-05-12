from typing import List
from ..abstract import Formatizer

"""
DEFAULT_FORMATIZER.PY
===============================================================
The default behaviour in which problems are parsed to queries 
to the LLMs. This format also determines how answers should be 
expected on the output, so a method for extracting them is 
also provided. Users may change the used template.
 
"""

class DefaultFormatizer(Formatizer):
    template = "Please solve the following problem: {statement};"

    def compile_problems(
        self,
        statements : List[str],
        **kwargs
    ) -> List[str]:
        keys = list(kwargs.keys())
        template = self.template
        row_count = len(statements)

        for key in keys:
            if key not in template:
                raise Exception(f'Passed key \'{key}\' but no {{{key}}} handle was found in current template')
            if len(kwargs[key]) != row_count:
                raise Exception(f'Number of elements in {key} must be the same as in statements ({row_count}). Current is {len(kwargs[key])}')
        
        output = []
        for i, statement in enumerate(statements):
            loc = "Please reason step by step, and put your final answer within \\boxed{} at the end of the solution. " + template
            loc = loc.replace('{statement}', statement)
            for key in keys:
                loc = loc.replace('{' + key + '}', kwargs[key][i])
            output.append(loc)
        return output

    @staticmethod
    def extract_answer(
        solution : str
    ) -> str:
        begin = solution.rfind("\\boxed{")
        end = solution.find("}", begin)
        if begin == -1 or end == -1:
            return None
        return solution[begin + len("\\boxed{"):end:]
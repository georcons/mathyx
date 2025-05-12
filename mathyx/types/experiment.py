from typing import List
from .result import Evaluation

class Experiment:
	_id : int = -1
	name : str
	description : str | None
	results : Evaluation

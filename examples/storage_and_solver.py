from mathyx import Solver, Storage

# Try this: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

"""
CREATE STORAGE

"""

problems = [
    'What is 1+2+3+...+99?',
    'What is the greatest prime number smaller than 100?',
    'Compute 8 times 7.',
    'Choose a random number from 1 to 100.'
]

storage = Storage.create('example storage')
storage.add_model('test model', 'vLLM', 'tiiuae/falcon-rw-1b', temperature=0.6, max_tokens=36000)
storage.add_problems(problems)
storage.add_problem_set('all', problems)



"""
EXAMPLE USAGE OF STORAGE AND SOLVER

"""

# Model from Storage
pipe_ = storage.model('test model')

# Problems from Storage
statements = storage.statements('all')

# Create solver
solver = Solver(pipeline = pipe_)

# Solve problems
result = solver.solve(statements, attempts=4)

# Display Results
print(result)

# Add results to Storage
storage.add_experiment('demo')
storage.add_results('demo', result)

# Inner storage pandas dataframe
print(storage._Storage__results)
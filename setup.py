from setuptools import setup, find_packages

def read_requirements():
    with open("./requirements/all_requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name = 'mathyx',
    version = "1.0.0",
    author = 'Nikolay Georgiev',
    description = 'A package for mathematical reasoning experiments and evaluation',
    packages = find_packages(),
    install_requires = read_requirements()
)
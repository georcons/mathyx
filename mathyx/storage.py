import pandas as pd
from typing import List
from datetime import datetime
from datasets import load_dataset, Dataset
from huggingface_hub import delete_repo
from .types import Model, Experiment, Pass, PromptResult, Evaluation

_RESERVED_ID_COLUMN = "__id"
_INIT_STR = "None"
_INIT_INT = -1

_STATEMENT_DS_SUFFIX = "_statements"
_STATEMENT_COLUMN = "statement"
_STATEMENT_KEYS = [_RESERVED_ID_COLUMN, _STATEMENT_COLUMN, "type", "answer", "domain", "custom_id", "solution"]
_STATEMENT_KEYS_SET = set(_STATEMENT_KEYS)
_STATEMENT_INIT = [_INIT_INT, _INIT_STR, _INIT_STR, _INIT_STR, _INIT_STR, _INIT_STR, _INIT_STR]

_EXPERIMENT_DS_SUFFIX = "_experiments"
_EXPERIMENT_KEYS = [_RESERVED_ID_COLUMN, "name", "description"]
_EXPERIMENT_KEYS_SET = set(_EXPERIMENT_KEYS)
_EXPERIMENT_INIT = [_INIT_INT, _INIT_STR, _INIT_STR]

_MODEL_DS_SUFFIX = "_models"
_MODEL_KEYS = [_RESERVED_ID_COLUMN, "name", "api_source", "model_name", "config"]
_MODEL_KEYS_SET = set(_MODEL_KEYS)
_MODEL_INIT = [_INIT_INT, _INIT_STR, _INIT_STR, _INIT_STR, _INIT_STR]

_RESULTS_DS_SUFFIX = "_results"
_RESULTS_KEYS = [_RESERVED_ID_COLUMN, "experiment_id", "model_id", "problem_id", "prompt", "model_solution", "date"]
_RESULT_KEYS_SET = set(_RESULTS_KEYS)
_RESULTS_INIT = [_INIT_INT, _INIT_INT, _INIT_INT, _INIT_INT, _INIT_STR, _INIT_STR, _INIT_STR]

_PROBLEM_SET_DS_SUFFIX = "_problem_sets"
_PROBLEM_SET_KEYS = [_RESERVED_ID_COLUMN, "name", "description", "problem_ids"]
_PROBLEM_SET_KEYS_SET = set(_PROBLEM_SET_KEYS)
_PROBLEM_SET_INIT = [_INIT_INT, _INIT_STR, _INIT_STR, _INIT_STR]

_CONFIG_DS_SUFFIX = "_config"
_CONFIG_KEYS = ["name", "description", "author", "created", "last_modified", "statement_id_state",
                "experiment_id_state", "model_id_state", "result_id_state", "problem_set_id_state"]
_CONFIG_KEYS_SET = set(_CONFIG_KEYS)



class Storage:
    name : str | None = "falcon.storage"
    description : str | None = None
    author : str | None = None
    created : datetime = datetime.now()
    last_modified : datetime = datetime.now()
    load_path : str | None = None
    __statements : pd.DataFrame
    __experiments : pd.DataFrame
    __models : pd.DataFrame
    __results : pd.DataFrame
    __problem_sets : pd.DataFrame
    __statement_id_state : int = 0
    __experiment_id_state : int = 0
    __model_id_state : int = 0
    __result_id_state : int = 0
    __problem_set_id_state : int = 0



    """
    1. CREATING STORAGES
    
    """
    def __init__(
        self,
        statements : pd.DataFrame,
        experiments : pd.DataFrame,
        models : pd.DataFrame,
        results : pd.DataFrame,
        problem_sets : pd.DataFrame,
        *,
        load_path : str | None = None,
        config : dict | None = None
    ):
        # Set load path
        self.load_path = load_path

        # Get key sets
        statement_keys = set(statements.columns)
        experiment_keys = set(experiments.columns)
        model_keys = set(models.columns)
        result_keys = set(results.columns)
        problem_set_keys = set(problem_sets.columns)

        # Validate no keys are missing
        missing_statement_keys = _STATEMENT_KEYS_SET - statement_keys
        missing_experiment_keys = _EXPERIMENT_KEYS_SET - experiment_keys
        missing_model_keys = _MODEL_KEYS_SET - model_keys
        missing_result_keys = _RESULT_KEYS_SET - result_keys
        missing_problem_set_keys = _PROBLEM_SET_KEYS_SET - problem_set_keys

        if len(missing_statement_keys) > 0:
            raise Exception(f'Keys missing from statements: {missing_statement_keys}')
        if len(missing_experiment_keys) > 0:
            raise Exception(f'Keys missing from experiments: {missing_experiment_keys}')
        if len(missing_model_keys) > 0:
            raise Exception(f'Keys missing from models: {missing_model_keys}')
        if len(missing_result_keys) > 0:
            raise Exception(f'Keys missing from result: {missing_result_keys}')
        if len(missing_problem_set_keys) > 0:
            raise Exception(f'Keys missing from problem sets: {missing_problem_set_keys}')

        # Validate there are no redundant keys
        redundant_statement_keys = statement_keys - _STATEMENT_KEYS_SET
        redundant_experiment_keys = experiment_keys - _EXPERIMENT_KEYS_SET
        redundant_model_keys = model_keys - _MODEL_KEYS_SET
        redundant_result_keys = result_keys - _RESULT_KEYS_SET
        redundant_problem_set_keys = problem_set_keys - _PROBLEM_SET_KEYS_SET

        if len(redundant_statement_keys) > 0:
            raise Exception(f'Unknown keys in statements: {redundant_statement_keys}')
        if len(redundant_experiment_keys) > 0:
            raise Exception(f'Unknown keys in experiments: {redundant_experiment_keys}')
        if len(redundant_model_keys) > 0:
            raise Exception(f'Unknown keys in models: {redundant_model_keys}')
        if len(redundant_result_keys) > 0:
            raise Exception(f'Unknown keys in result: {redundant_result_keys}')
        if len(redundant_problem_set_keys) > 0:
            raise Exception(f'Unknown keys in problem sets: {redundant_problem_set_keys}')

        # Validate config if present and write its data to the object
        if config is not None:
            config_keys = set(config.keys())
            missing_config_keys = _CONFIG_KEYS_SET - config_keys
            redundant_config_keys = config_keys - _CONFIG_KEYS_SET
            if len(missing_config_keys) > 0:
                raise Exception(f'Config is missing keys {missing_config_keys}')
            if len(redundant_config_keys) > 0:
                raise Exception(f'Unknown keys in config: {redundant_config_keys}')
            self.name = config['name']
            self.description = config['description']
            self.author = config['author']
            self.created = config['created']
            self.last_modified = config['last_modified']
            self.__statement_id_state = config['statement_id_state']
            self.__experiment_id_state = config['experiment_id_state']
            self.__model_id_state = config['model_id_state']
            self.__result_id_state = config['result_id_state']
            self.__problem_set_id_state = config['problem_set_id_state']

        else:
            # Otherwise retrieve id states from tables
            self.__statements_id_state = self.__id_state(statements) + 1
            self.__experiment_id_state = self.__id_state(experiments) + 1
            self.__model_id_state = self.__id_state(models) + 1
            self.__result_id_state = self.__id_state(results) + 1
            self.__problem_set_id_state = self.__id_state(problem_sets) + 1

        # Save dataframes to object
        self.__statements = statements
        self.__experiments = experiments
        self.__models = models 
        self.__results = results
        self.__problem_sets = problem_sets

    @staticmethod
    def create(
        name : str,
        *,
        description : str | None = None,
        author : str | None = None
    ):
        statements = pd.DataFrame({key: [val] for key, val in zip(_STATEMENT_KEYS, _STATEMENT_INIT)})
        experiments = pd.DataFrame({key: [val] for key, val in zip(_EXPERIMENT_KEYS, _EXPERIMENT_INIT)})
        models = pd.DataFrame({key: [val] for key, val in zip(_MODEL_KEYS, _MODEL_INIT)})
        results = pd.DataFrame({key: [val] for key, val in zip(_RESULTS_KEYS, _RESULTS_INIT)})
        problem_sets = pd.DataFrame({key: [val] for key, val in zip(_PROBLEM_SET_KEYS, _PROBLEM_SET_INIT)})
        config = {
            "name" : name,
            "description" : description,
            "author" : author,
            "created" : datetime.now(),
            "last_modified" : datetime.now(),
            "statement_id_state" : 0,
            "experiment_id_state" : 0,
            "model_id_state" : 0,
            "result_id_state" : 0,
            "problem_set_id_state" : 0   
        }

        return Storage(
            statements,
            experiments,
            models,
            results,
            problem_sets,
            load_path = None,
            config = config
        )

    # Modification Decoration
    def modify(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.last_modified = datetime.now()
            return result
        return wrapper

    """
    2. PROBLEMS
    
    """
    def problems(
        self,
        set_name : str | None = None
    ) -> pd.DataFrame:
        if set_name is None:
            return self.__statements[1:]
        problem_set = self.__get_problem_set(set_name)
        if problem_set is None:
            raise Exception(f'No Problem Set named \'{set_name}\' found in storage.')
        pids = [int(_id) for _id in problem_set['problem_ids'].split(',')]
        return self.__statements[self.__statements[_RESERVED_ID_COLUMN].isin(pids)]

    def statements(
        self,
        set_name : str | None = None
    ) -> List[str]:
        return self.problems(set_name)[_STATEMENT_COLUMN].to_list()

    def answers(
        self,
        set_name : str | None = None
    ) -> List[str]:
        return self.problems(set_name)['answer'].to_list()

    @modify
    def add_problems(
        self,
        statements : List[str],
        **kwargs
    ) -> None:
        row_count = len(statements)
        dictionary = {k : [str(elem) for elem in v] for k,v in kwargs.items()}
        dictionary[_STATEMENT_COLUMN] = statements
        dictionary[_RESERVED_ID_COLUMN] = [i + self.__statement_id_state for i in range(row_count)]
        df = pd.DataFrame(dictionary)
        df_keys = set(df.columns)
        redundant_keys = df_keys - _STATEMENT_KEYS_SET
        if len(redundant_keys) > 0:
            raise Exception(f'Invalid keys: {redundant_keys}')
        new_df = pd.concat([self.__statements, df], ignore_index=True)
        self.__statements = new_df.drop_duplicates(subset=_STATEMENT_COLUMN, keep='first')
        self.__statement_id_state += row_count


    # !UNSAFE
    @modify
    def remove_problems(
        self,
        statements : List[str]
    ) -> None:
        current_df = self.__statements
        self.__statements = current_df[~current_df[_STATEMENT_COLUMN].isin(statements)]
    


    """
    3. HANDLING MODELS

    """
    @property
    def models(self) -> List[str]:
        return list(self.__models[1:]['name'].values)

    @property
    def models_pd(self) -> pd.DataFrame:
        return self.__models[1:]

    def model(
        self,
        name : str
    ) -> Model:
        df = self.__models
        if name not in df['name'].values:
            raise Exception(f'Invalid model name: {name}')
        row = df[df['name'] == name].iloc[0]
        model = Model()
        model._id = int(row[_RESERVED_ID_COLUMN])
        model.name = row['name']
        model.source = row['api_source']
        model.model = row['model_name']
        model.config = self.__parse_config(row['config'])
        return model

    @modify
    def add_model(
        self,
        name : str,
        api_source : str | None,
        model_name : str | None,
        **kwargs
    ) -> None:
        if name in self.__models['name'].values:
            raise Exception('Model with that name already exists in the storage')
        config_str = self.__model_config_to_str(kwargs)
        new_row = pd.DataFrame([{
            _RESERVED_ID_COLUMN: self.__model_id_state,
            'name': name,
            'api_source': api_source, 
            'model_name': model_name,
            'config': config_str
        }])
        self.__models = pd.concat([self.__models, new_row], ignore_index=True)
        self.__model_id_state += 1

    @modify
    def remove_model(
        self,
        name : str
    ) -> None:
        current_df = self.__models
        mid = self.__get_model(name)[_RESERVED_ID_COLUMN]
        self.__models = current_df[current_df['name'] != name]
        self.__results = self.__results[self.__results['model_id'] != mid]
    


    """
    4. EXPERIMENTS AND RESULTS

    """
    @property
    def experiments(self) -> List[str]:
        return list(self.__experiments['name'][1:].values)

    @modify
    def experiment(
        self,
        name : str,
        *,  
        # Optional filters
        set_name : str | None = None,
        model : str | None = None,
        statements : List[str] | None = None
    ) -> Experiment:
        experiment = self.__get_experiment(name)
        if experiment is None:
            raise Exception(f'Experiment \'{name}\' not found in storage')
        eid = experiment[_RESERVED_ID_COLUMN]
        description = experiment['description']
        filtered_results = self.__results[self.__results['experiment_id'] == eid]

        if model is not None:
            mod = self.__get_model(model)
            if mod is None:
                raise Exception(f'Model \'{model}\' not found in storage')
            mid = mod[_RESERVED_ID_COLUMN]
            filtered_results = filtered_results[filtered_results['model_id'] == mid]
        
        if statements is not None:
            # if set is not None:
            #     raise Exception('Cannot parse both \'statements\' and \'set\' argument')
            filtered_statements = self.__statements[self.__statements[_STATEMENT_COLUMN].isin(statements)]
            sids = filtered_statements[_RESERVED_ID_COLUMN].to_list()
            filtered_results = filtered_results[filtered_results['problem_id'].isin(sids)]
        
        if set_name is not None:
            problem_set = self.__get_problem_set(set_name)
            if problem_set is None:
                raise Exception(f'Problem set \'{set_name}\' not found in storage')
            sids = [int(_id) for _id in problem_set['problem_ids'].split(',')]
            filtered_results = filtered_results[filtered_results['problem_id'].isin(sids)]
        
        prompts = filtered_results['prompt'].drop_duplicates().to_list()

        prompt_results = []

        for prompt in prompts:
            rows = filtered_results[filtered_results['prompt'] == prompt]
            if len(rows) != 0:
                pid = rows['problem_id'].iloc[0]
                statements = self.__statements[self.__statements[_RESERVED_ID_COLUMN] == pid]
                statement = None
                if len(statements) != 0:
                    statement = statements.iloc[0]
                responses = rows['model_solution'].to_list()
                prompt_res = PromptResult(responses, prompt=prompt)
                prompt_res.set_statement(statement[_STATEMENT_COLUMN])
                if 'answer' in statement:
                    prompt_res.correct_answer = statement['answer']
                prompt_results.append(prompt_res)

        eval_ = Evaluation(prompt_results)

        output = Experiment()
        output._id = eid
        output.name = name
        output.description = description
        output.results = eval_
        return output

    @modify
    def add_experiment(
        self,
        name : str,
        *,
        description : str | None = None
    ) -> None:
        if name in self.__experiments['name'].values:
            raise Exception('Experiment with that name already exists in the storage')
        new_row = pd.DataFrame([{
            _RESERVED_ID_COLUMN: self.__experiment_id_state,
            'name' : name,
            'description' : description
        }])
        self.__experiments = pd.concat([self.__experiments, new_row], ignore_index=True)
        self.__experiment_id_state += 1

    @modify
    def remove_experiment(
        self,
        name : str,
        *,
        confirm : bool | None = False
    ) -> None:
        if not confirm:
            raise Exception('To remove experiment data, confirm by setting argument confirm=True.')
        experiment = self.__get_experiment(name)
        if experiment is None:
            raise Exception(f'Experiment \'{name}\' not found in storage')
        eid = experiment[_RESERVED_ID_COLUMN]
        self.__results = self.__results[self.__results['experiment_id'] != eid]
        self.__experiments = self.__experiments[self.__experiments[_RESERVED_ID_COLUMN] != eid]

    @modify
    def add_results(
        self,
        experiment : str,
        problems : List[Pass] | Evaluation,
        *,
        model : str | None = None
    ) -> None:
        if isinstance(problems, Evaluation):
            problems = self.__parse_results(problems)
        exp = self.__get_experiment(experiment)
        if exp is None:
            raise Exception(f'Experiment \'{experiment}\' not found in storage')
        eid = exp[_RESERVED_ID_COLUMN]
        mid = None
        if model is not None:
            mod = self.__get_model(model)
            if model is None:
                raise Exception(f'Model \'{model}\' not found in storage')
            mid = mod[_RESERVED_ID_COLUMN]
        count = len(problems)
        rids = list(range(self.__result_id_state, self.__result_id_state + count))
        pids = []
        for pass_ in problems:
            _statement = self.__get_problem(pass_.statement)
            if _statement is None:
                raise Exception(f'Statement \'{pass_.statement}\' not found in the storage.')
            pids.append(_statement[_RESERVED_ID_COLUMN])
        new_rows = pd.DataFrame({
            _RESERVED_ID_COLUMN: rids,
            'problem_id': pids,
            'prompt': [pass_.user for pass_ in problems],
            'model_solution': [pass_.assistant for pass_ in problems],
            'date': [pass_.date for pass_ in problems]
        })
        new_rows['experiment_id'] = eid
        new_rows['model_id'] = mid
        self.__results = pd.concat([self.__results, new_rows], ignore_index=True)
        self.__result_id_state += count
    
    @modify
    def remove_results(
        self,
        experiment : str,
        *,
        model : str | None = None,
        set : str | None = None
    ) -> None:
        raise NotImplementedError



    """
    5. PROBLEM SETS

    """
    @property
    def problem_sets(
        self
    ) -> List[str]:
        return list(self.__problem_sets[1:]['name'].values)
    
    @modify
    def add_problem_set(
        self,
        name : str,
        statements : List[str],
        *,
        description : str | None = None
    ) -> None:
        statement_rows = self.__statements[self.__statements[_STATEMENT_COLUMN].isin(statements)]
        sids = [str(_id) for _id in statement_rows[_RESERVED_ID_COLUMN].to_list()]
        new_row = pd.DataFrame([{
            _RESERVED_ID_COLUMN: self.__problem_set_id_state,
            'name': name,
            'description': description,
            'problem_ids': ','.join(sids)
        }])
        self.__problem_set_id_state += 1
        self.__problem_sets = pd.concat([self.__problem_sets, new_row], ignore_index=True)

    @modify
    def remove_problem_set(
        self,
        name : str,
        *,
        keep_problems : bool | None = True
    ) -> None:
        current_df = self.__problem_sets
        problem_set = self.__get_problem_set(name)
        if problem_set is None:
            raise Exception(f'No problem set \'{name}\' in storage.')
        _id = problem_set[_RESERVED_ID_COLUMN]
        pids = [int(_id) for _id in problem_set['problem_ids'].split(',')]
        self.__problem_sets = current_df[current_df['name'] != name]
        if not keep_problems:
            self.__results = self.__results[~self.__results['problem_id'].isin(pids)]


    """
    6. PUBLISHING, LOADING AND REFRESHING

    """
    def push_to_hub(
        self,
        hf_path : str | None = None,
        /,
        token : str | None = None
    ) -> None:
        if hf_path is None:
            if self.load_path is None:
                raise Exception('You must specify a HF path for this storage.')
            else: return self.push_to_hub(self.load_path, token=token)

        config_df = self.__storage_config_to_df()
        dataframes = [config_df, self.__statements, self.__experiments, self.__models, self.__results, self.__problem_sets]
        suffixes = [_CONFIG_DS_SUFFIX, _STATEMENT_DS_SUFFIX, _EXPERIMENT_DS_SUFFIX, _MODEL_DS_SUFFIX, _RESULTS_DS_SUFFIX, _PROBLEM_SET_DS_SUFFIX]

        for df, suffix in zip(dataframes, suffixes):
            ds = Dataset.from_pandas(df)
            ds.push_to_hub(hf_path + suffix, token=token)

    @staticmethod
    def load_storage(
        hf_path : str,
        /,
        token : str | None = None
    ):
        config_ = load_dataset(hf_path +  _CONFIG_DS_SUFFIX, token=token)['train'].to_dict()
        config = {k: v[0] for k,v in config_.items()}
        statements = load_dataset(hf_path + _STATEMENT_DS_SUFFIX, token=token)['train'].to_pandas()
        experiments = load_dataset(hf_path + _EXPERIMENT_DS_SUFFIX, token=token)['train'].to_pandas()
        models = load_dataset(hf_path + _MODEL_DS_SUFFIX, token=token)['train'].to_pandas()
        results = load_dataset(hf_path + _RESULTS_DS_SUFFIX, token=token)['train'].to_pandas()
        problem_sets = load_dataset(hf_path + _PROBLEM_SET_DS_SUFFIX, token=token)['train'].to_pandas()
        return Storage(
            statements,
            experiments,
            models,
            results,
            problem_sets,
            load_path=hf_path,
            config=config
        )

    def refresh(
        self
    ):
        if self.load_path is None:
            raise Exception('This storage is locally created. Only storages loaded from Hugging Face can be refreshed.')
        temp_ = self.__class__.load_storage(self.load_path)
        self.__dict__.update(temp_.__dict__)

    """
    7. DELETING STORAGES

    """

    @staticmethod
    def delete_storage(
        storage_path : str,
        *,
        token : str | None = None,
        confirm : bool | None = False
    ):
        if not confirm:
            raise Exception('To delete a storage, set argument confirm=True. WARNING: This operation is irreversible.')
        suffixes = [_CONFIG_DS_SUFFIX, _STATEMENT_DS_SUFFIX, _EXPERIMENT_DS_SUFFIX, _MODEL_DS_SUFFIX, _RESULTS_DS_SUFFIX, _PROBLEM_SET_DS_SUFFIX]
        for suffix in suffixes:
            repo_id = storage_path + suffix
            try: delete_repo(repo_id=repo_id, repo_type='dataset', token=token)
            except: print(f"ERROR: Internal dataset {repo_id} not found or you don't have access to delete it.")
        

    """
    A. Private Methods

    """

    def __storage_config_to_df(self):
        return pd.DataFrame({
            'name' : [self.name],
            'description' : [self.description],
            'author' : [self.author],
            'created' : [self.created],
            'last_modified' : [self.last_modified],
            'statement_id_state' : [self.__statement_id_state],
            'experiment_id_state' : [self.__experiment_id_state],
            'model_id_state' : [self.__model_id_state],
            'result_id_state' : [self.__result_id_state],
            'problem_set_id_state' : [self.__problem_set_id_state] 
        })

    def __get_problem_set(
        self,
        name : str
    ) -> pd.Series | None:
        if name not in self.__problem_sets['name'].values:
            return None
        return self.__problem_sets[self.__problem_sets['name'] == name].iloc[0]

    def __parse_results(
        self,
        result_list : Evaluation
    ) -> List[Pass]:
        output = []
        for prompt_result in result_list:
            for pass_ in prompt_result:
                output.append(pass_)
        return output

    def __get_problem(
        self,
        statement : str
    ) -> pd.Series | None:
        if statement not in self.__statements[_STATEMENT_COLUMN].values:
            return None
        return self.__statements[self.__statements[_STATEMENT_COLUMN] == statement].iloc[0]

    def __get_model(
        self,
        name : str
    ) -> pd.Series | None:
        if name not in self.__models['name'].values:
            return None
        return self.__models[self.__models['name'] == name].iloc[0]

    def __get_experiment(
        self,
        name : str
    ) -> pd.Series | None:
        if name not in self.__experiments['name'].values:
            return None
        return self.__experiments[self.__experiments['name'] == name].iloc[0]

    def __model_config_to_str(
        self,
        params : dict
    ) -> str:
        lst = []
        for key in list(params.keys()):
            val = params[key]
            if isinstance(val, float):
                lst.append(":".join([key, "float", str(val)]))
            elif isinstance(val, int):
                lst.append(":".join([key, "int", str(val)]))
            elif isinstance(val, bool):
                lst.append(":".join([key, "bool", str(val)]))
            elif isinstance(val, str):
                lst.append(":".join([key, "str", val]))
            else:
                raise Exception(f"Invalid value of key '{key}': {val}")
        return ";".join(lst)

    def __parse_config(
        self,
        config : str
    ) -> dict:
        blocks = config.split(';')
        output = {}
        for block in blocks:
            subblocks = block.split(':')
            if len(subblocks) != 3:
                continue
            key = subblocks[0]
            type_name = subblocks[1]
            value_str = subblocks[2]
            value = None
            try:
                if type_name == 'float':
                    value = float(value_str)
                elif type_name == 'int':
                    value = int(value_str)
                elif type_name == 'bool':
                    value = bool(value_str)
                elif type_name == 'str':
                    value = value_str
                else:
                    continue
            except:
                continue
            output[key] = value
        return output

    def __id_state(self, table):
        return -1 if (len(table[_RESERVED_ID_COLUMN]) == 0 or table[_RESERVED_ID_COLUMN][-1] == -1) else table[_RESERVED_ID_COLUMN][-1]

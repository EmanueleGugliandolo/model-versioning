# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:28:20 2022

@author: 08500217
"""
from typing import Dict, Any, Tuple, Callable, List, Optional, IO
import sys
import spacy
from spacy import util
from spacy import Language
from spacy.training.loggers import console_logger
import collections
import pandas as pd

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@spacy.registry.loggers("mlflow_logger.v1")
def mlflow_logger(
    project_name: str,
    remove_config_values: List[str] = [],
    model_log_interval: Optional[int] = None,
    log_dataset_dir: Optional[str] = None,
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
    log_best_dir: Optional[str] = None,
    log_latest_dir: Optional[str] = None,
):
    try:
        import mlflow
        print("mlflow imported correctly")
    except ImportError:
        raise ImportError(
            "The 'mlflow' library could not be found - did you install it? "
        )
    console = console_logger(progress_bar=False)
    
    def setup_logger(
        nlp: "Language", stdout: IO = sys.stdout, stderr: IO = sys.stderr
    ) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        
        flattened = flatten(nlp.config)
        d = {k:v for k,v in flattened.items() if "@" not in k}
        
        today_time = str(pd.Timestamp.today().round(freq='S'))
        default_run_name = "run_" + today_time.split()[0] + "_" + today_time.split()[1]
        
        mlflow.set_experiment(project_name)
        mlflow.start_run(run_name=default_run_name)
        mlflow.log_params(d)
        
        config = nlp.config.interpolate()
        config_dot = util.dict_to_dot(config)
        for field in remove_config_values:
            del config_dot[field]
        config = util.dot_to_dict(config_dot)

        console_log_step, console_finalize = console(nlp, stdout, stderr)

        
        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            
            if info is not None:
                score = info["score"]
                other_scores = info["other_scores"]
                losses = info["losses"]
                mlflow.log_metrics({"score": score})
                if losses:
                    mlflow.log_metrics({f"loss_{k}": v for k, v in losses.items()})
                if isinstance(other_scores, dict):
                    mlflow.log_metrics(flatten(other_scores))

        def finalize() -> None:

            console_finalize()
        return log_step, finalize

    mlflow.end_run()
    return setup_logger            



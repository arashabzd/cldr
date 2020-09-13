# this code is borrowed from 
# https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/local_evaluation.py
import os
import disentanglement_lib
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1

import evaluate
disentanglement_lib.evaluation.evaluate = evaluate


def compute_metrics(model_dir, 
                    output_dir,
                    dataset_name,
                    cuda=False,
                    metrics=None, 
                    overwrite=True):
    _study = unsupervised_study_v1.UnsupervisedStudyV1()
    evaluation_configs = sorted(_study.get_eval_config_files())
    
    if metrics is None:
        metrics = [
            'beta_vae_sklearn'
            'dci',
            'downstream_task_boosted_trees',
            'downstream_task_logistic_regression',
            'factor_vae_metric',
            'irs',
            'mig',
            'modularity_explicitness',
            'sap_score',    
            'unsupervised'
        ]
    
    for gin_eval_config in evaluation_configs:
        metric = gin_eval_config.split("/")[-1].replace(".gin", "")
        if  metric not in metrics:
            continue
        
        print("Evaluating Metric: {}".format(metric))
        bindings = [
            "evaluation.random_seed = {}".format(0),
            "evaluation.name = '{}'".format(metric)
        ]
        metric_dir = os.path.join(output_dir, metric)
        evaluate.evaluate_with_gin(
            model_dir,
            metric_dir,
            dataset_name,
            cuda,
            overwrite,
            [gin_eval_config],
            bindings
        )

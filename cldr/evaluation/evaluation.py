# this code is borrowed from 
# https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/local_evaluation.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import disentanglement_lib
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1

from . import evaluate
disentanglement_lib.evaluation.evaluate = evaluate


def compute_metrics(model_path, 
                    output_dir,
                    dataset_name,
                    cuda=False,
                    metrics=None,
                    seed=0,
                    overwrite=True):
    _study = unsupervised_study_v1.UnsupervisedStudyV1()
    evaluation_configs = sorted(_study.get_eval_config_files())
    
    if metrics is None:
        metrics = [
            'dci',
            'sap_score',
            'irs',
            'downstream_task_boosted_trees',
            'downstream_task_logistic_regression',
            'beta_vae_sklearn',
            'factor_vae_metric',
            'mig',
            'modularity_explicitness',
            'unsupervised'
        ]
    
    for gin_eval_config in evaluation_configs:
        metric = gin_eval_config.split("/")[-1].replace(".gin", "")
        if  metric not in metrics:
            continue
        
        print("Evaluating Metric: {}".format(metric))
        bindings = [
            "evaluation.random_seed = {}".format(seed),
            "evaluation.name = '{}'".format(metric)
        ]
        metric_dir = os.path.join(output_dir, metric)
        evaluate.evaluate_with_gin(
            model_path,
            metric_dir,
            dataset_name,
            cuda,
            overwrite,
            [gin_eval_config],
            bindings
        )

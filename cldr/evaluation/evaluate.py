# this code is borrowed from 
# https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/evaluate.py

import os
import time

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.utils import results
import numpy as np
import tensorflow as tf
import gin.tf

from ..utils import utils
from tensorflow.python.framework.errors_impl import NotFoundError


def evaluate_with_gin(model_path,
                      output_dir,
                      dataset_name,
                      cuda,
                      overwrite,
                      gin_config_files=None,
                      gin_bindings=None):
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    evaluate(model_path, output_dir, dataset_name, cuda, overwrite)
    gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_path", "output_dir", "dataset_name", "cuda", "overwrite"])
def evaluate(model_path,
             output_dir,
             dataset_name,
             cuda,
             overwrite,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):

    if tf.gfile.IsDirectory(output_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(output_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")

    if gin.query_parameter("dataset.name") == "auto":
        with gin.unlock_config():
            gin.bind_parameter("dataset.name", dataset_name)
    dataset = named_data.get_named_ground_truth_data()

    experiment_timer = time.time()
    if os.path.exists(model_path):
        results_dict = _evaluate_with_pytorch(model_path, evaluation_fn,
                                              dataset, cuda, random_seed)
    else:
        raise RuntimeError("`model_dir` must contain either a pytorch or a TFHub model.")

    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(output_dir, "evaluation", results_dict)


def _evaluate_with_pytorch(model_path, evalulation_fn, dataset, cuda, random_seed):
    model = utils.import_model(model_path)
    _representation_function = utils.make_representor(model, cuda)
    results_dict = evalulation_fn(
        dataset,
        _representation_function,
        random_state=np.random.RandomState(random_seed)
    )
    return results_dict

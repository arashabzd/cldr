# this code is borrowed from 
# https://github.com/AIcrowd/neurips2019_disentanglement_challenge_starter_kit/blob/master/evaluate.py

"""Evaluation protocol to compute metrics."""
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

import utils
from tensorflow.python.framework.errors_impl import NotFoundError


def evaluate_with_gin(model_dir,
                      output_dir,
                      dataset_name,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
    """Evaluate a representation based on the provided gin configuration.

    This function will set the provided gin bindings, call the evaluate()
    function and clear the gin config. Please see the evaluate() for required
    gin bindings.

    Args:
      model_dir: String with path to directory where the representation is saved.
      output_dir: String with the path where the evaluation should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    evaluate(model_dir, output_dir, dataset_name, overwrite)
    gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_dir", "output_dir", "dataset_name", "overwrite"])
def evaluate(model_dir,
             output_dir,
             dataset_name,
             overwrite=False,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name=""):
    """Loads a representation TFHub module and computes disentanglement metrics.

    Args:
      model_dir: String with path to directory where the representation function
        is saved.
      output_dir: String with the path where the results should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      evaluation_fn: Function used to evaluate the representation (see metrics/
        for examples).
      random_seed: Integer with random seed used for training.
      name: Optional string with name of the metric (can be used to name metrics).
    """
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
    if os.path.exists(os.path.join(model_dir, 'model.pt')):
        model_path = os.path.join(model_dir, 'model.pt')
        results_dict = _evaluate_with_pytorch(model_path, evaluation_fn,
                                              dataset, random_seed)
    else:
        raise RuntimeError("`model_dir` must contain either a pytorch or a TFHub model.")

    # Save the results (and all previous results in the pipeline) on disk.
    original_results_dir = os.path.join(model_dir, "results")
    results_dir = os.path.join(output_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(results_dir, "evaluation", results_dict,
                                    original_results_dir)


def _evaluate_with_pytorch(model_path, evalulation_fn, dataset, random_seed):
    model = utils.import_model(model_path)
    _representation_function = utils.make_representor(model)
    results_dict = evalulation_fn(
        dataset,
        _representation_function,
        random_state=np.random.RandomState(random_seed)
    )
    return results_dict

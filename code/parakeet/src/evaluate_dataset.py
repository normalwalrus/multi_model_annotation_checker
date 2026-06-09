import logging
import os
import sys

import yaml

sys.path.append("/asr-eval")

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

from evaluate_utils.evaluation_utils import (
    individual_entry_evaluation,
    overall_evaluation,
)
from evaluate_utils.general_utils import (
    add_dict_to_json_file,
    list_of_dicts_to_json_file,
)

from whitebox.parakeet.src.transcribe import Canary1bEvaluator

config = yaml.safe_load(open("/asr-eval/evaluate_dataset/config.yml"))

MODEL_NAME = "canary-1b-flash"
DATASET_NAME = config["dataset"]["name"]
PATH_TO_DATASET_FOLDER = config["dataset"]["dataset_folder"] + DATASET_NAME

ENTRIES_TO_RUN = config["models"][MODEL_NAME]["entries_to_run"]
ASR_MODEL_PATH = config["models"][MODEL_NAME]["asr_model_path"]
KENLM_MODEL_PATH = config["models"][MODEL_NAME]["kenlm_model_path"]
OUTPUT_MANIFEST_FOLDER = config["dataset"]["evaluation_results_folder"] + MODEL_NAME


def individual_entry_evaluation_script() -> list:
    """
    Takes the individual_entry_evaluation() function and output the json files evaluation for each dataset in DATASETS

    Returns:
    List of dictionaries: Metadata that has been created for reference (And functions like overall_evaluation_script()).
    """
    evaluator = Canary1bEvaluator(ASR_MODEL_PATH)

    print(f"Starting evaluation for dataset: {DATASET_NAME}")
    # Function for the evaluation of model based on the dataset name (Assumed it is in /data)
    individual_entry_evaluation_list, time_elasped, channels = (
        individual_entry_evaluation(
            evaluator=evaluator, dataset=DATASET_NAME, ENTRIES_TO_RUN=ENTRIES_TO_RUN
        )
    )

    # Create output manifest folder if not existing
    if not os.path.exists(OUTPUT_MANIFEST_FOLDER):
        os.mkdir(OUTPUT_MANIFEST_FOLDER)

    # Output each individual entry's transciption and reference with respective evaluation as json file
    path_to_json = (
        OUTPUT_MANIFEST_FOLDER
        + "/"
        + MODEL_NAME
        + "_"
        + DATASET_NAME
        + "_individual_evaluation.json"
    )
    list_of_dicts_to_json_file(individual_entry_evaluation_list, path_to_json)

    # Prepare metadata (Used in overall_evaluation_script())
    metadata_entry = {
        "dataset": DATASET_NAME,
        "time": time_elasped,
        "path_to_json": path_to_json,
        "channels": channels,
    }

    return metadata_entry


def overall_evaluation_script(metadata_entry: list):
    """
    Takes the overall_evaluation() function and output the txt file for overall evaluation for each dataset in DATASETS

    Parameters:
    metadata_list (List of Dictionaries): Metadata that has been created in each element in the list:
        dataset(str): name of dataset
        time(float): time elasped to run dataset through evaluator
        path_to_json(str): path to the json file containing the individual entry evaluation

    """

    dataset = metadata_entry["dataset"]
    time_elasped = metadata_entry["time"]
    path_to_json = metadata_entry["path_to_json"]
    channels = metadata_entry["channels"]

    # Using overall_evaluation() function to do the evaluation and produce a dictionary
    overall_evaluation_entry = overall_evaluation(
        dataset=dataset,
        time_elasped=time_elasped,
        path_to_json=path_to_json,
        model=MODEL_NAME,
        channels=channels,
    )

    # Create the final json file for evaluation
    save_path_overall_evalution = (
        OUTPUT_MANIFEST_FOLDER + "/" + MODEL_NAME + "_overall_evaluation.json"
    )
    add_dict_to_json_file(overall_evaluation_entry, save_path_overall_evalution)


if __name__ == "__main__":

    metadata_entry = individual_entry_evaluation_script()

    overall_evaluation_script(metadata_entry=metadata_entry)

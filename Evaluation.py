import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
import wandb
from model.Utils import replicability, Config, hamming_score, rounder


def evaluation_AP(prediction_path=None, label_path=None, mlb=None):

    prediction_list = []
    label_list = []

    id2label = {}
    conversations = torch.load(label_path)
    for conversation in conversations:
        for example in conversation:
            id2label[example["example_id"]] = list(
                mlb.fit_transform([example["system_action"]])[0]
            )

    with open(prediction_path, "r") as r:
        for line in r:
            if len(line.rstrip().split()) == 1:
                actions = []
            else:
                example_id, prediction = line.rstrip().split("\t")
                actions = prediction.split(",")

            prediction_list.append(list(mlb.fit_transform([actions])[0]))
            label_list.append(id2label[example_id])

    print(len(id2label), len(prediction_list))
    assert len(id2label) == len(prediction_list)

    f1 = f1_score(label_list, prediction_list, average="macro")
    precision = precision_score(label_list, prediction_list, average="macro")
    recall = recall_score(label_list, prediction_list, average="macro")
    acc_exact = accuracy_score(label_list, prediction_list)
    acc_hamming = hamming_score(np.array(label_list), np.array(prediction_list))
    jaccard = jaccard_score(
        np.array(label_list), np.array(prediction_list), average="samples"
    )
    precision_detail = precision_score(label_list, prediction_list, average=None)
    recall_detail = recall_score(label_list, prediction_list, average=None)

    prediction_sys_action_num_all_turns = [
        sum(prediction) for prediction in prediction_list
    ]

    result_dict = {
        "f1": rounder(f1),
        "p": rounder(precision),
        "r": rounder(recall),
        "jaccard": rounder(jaccard),
        "acc_hamming": rounder(acc_hamming),
        "acc_exact": rounder(acc_exact),
        "aver_sys_action_num": sum(prediction_sys_action_num_all_turns)
        / len(prediction_sys_action_num_all_turns),
        "min_sys_action_num": min(prediction_sys_action_num_all_turns),
        "max_sys_action_num": max(prediction_sys_action_num_all_turns),
        "p_per_label": [rounder(i) for i in precision_detail],
        "r_per_label": [rounder(i) for i in recall_detail],
    }
    print(result_dict)
    return result_dict


def evaluation_SIP(prediction_path=None, label_path=None):
    prediction_list = []
    label_list = []

    id2label = {}
    conversations = torch.load(label_path)
    for conversation in conversations:
        for example in conversation:
            id2label[example["example_id"]] = example["system_I_label"]

    with open(prediction_path, "r") as r:
        for line in r:
            example_id, prediction = line.rstrip().split("\t")
            prediction_list.append(prediction)
            label_list.append(id2label[example_id])

    assert len(id2label) == len(prediction_list)

    label_list = [1 if i == "Initiative" else 0 for i in label_list]
    prediction_list = [1 if i == "Initiative" else 0 for i in prediction_list]

    acc = accuracy_score(label_list, prediction_list)
    matrix = confusion_matrix(label_list, prediction_list, labels=[0, 1])
    acc_per_label = matrix.diagonal() / matrix.sum(axis=1)
    total_num = matrix.sum(axis=1).tolist()
    hit_num = matrix.diagonal().tolist()
    f1 = f1_score(label_list, prediction_list, average="macro")
    precision = precision_score(label_list, prediction_list, average="macro")
    recall = recall_score(label_list, prediction_list, average="macro")
    precision_detail = precision_score(label_list, prediction_list, average=None)
    recall_detail = recall_score(label_list, prediction_list, average=None)

    result_dict = {
        "f1": rounder(f1),
        "p": rounder(precision),
        "r": rounder(recall),
        "acc": rounder(acc),
        "acc_per_label": [rounder(i) for i in acc_per_label],
        "total_num": total_num,
        "hit_num": hit_num,
        "p_per_label": [rounder(i) for i in precision_detail],
        "r_per_label": [rounder(i) for i in recall_detail],
    }

    print(result_dict)
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--epoch_num", type=int, default=20)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ikat",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or user). Defaults to your personal account.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run W&B in offline mode (sync later with `wandb sync`)",
    )

    args = parser.parse_args()

    if "WISE" in args.label_path:
        dataset = "WISE"
    elif "MSDialog" in args.label_path:
        dataset = "MSDialog"
    elif "ClariQ" in args.label_path:
        dataset = "ClariQ"
    else:
        raise NotImplementedError

    if "train" in args.label_path:
        args.dataset_type = "train"
    elif "valid" in args.label_path:
        args.dataset_type = "valid"
    elif "test" in args.label_path:
        args.dataset_type = "test"
    else:
        raise NotImplementedError

    config = Config(args)
    mlb = MultiLabelBinarizer(classes=config.action)

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    run_name = args.wandb_run_name or f"eval-{dataset}"
    task_tag = "AP" if "AP" in args.prediction_path else "SIP"

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        tags=[dataset, task_tag, "evaluation"],
        job_type="evaluation",
    )

    label_artifact = wandb.Artifact(
        name="labels",
        type="dataset",
        description=f"{dataset} ground-truth labels",
        metadata={"dataset": dataset},
    )
    label_artifact.add_file(args.label_path)
    run.log_artifact(label_artifact)

    pred_artifact = wandb.Artifact(
        name="predictions",
        type="predictions",
        description=f"{dataset} model predictions",
        metadata={"dataset": dataset, "task": task_tag},
    )
    pred_artifact.add_dir(args.prediction_path)
    run.log_artifact(pred_artifact)

    for epoch_id in range(0, args.epoch_num + 1):
        prediction_path_ = (
            args.prediction_path
            + "/"
            + args.dataset_type
            + "."
            + str(epoch_id)
            + ".txt"
        )

        if os.path.exists(prediction_path_):
            print(f"Start to evaluate {epoch_id}")
            if "AP" in args.prediction_path:
                result_dict = evaluation_AP(prediction_path_, args.label_path, mlb)
            else:
                result_dict = evaluation_SIP(prediction_path_, args.label_path)

            with open(
                args.prediction_path + "/" + "result.+" + args.dataset_type + "+.txt",
                "a+",
                encoding="utf-8",
            ) as w:
                w.write(str(epoch_id) + ": " + str(result_dict) + os.linesep)

            flat_metrics = {"epoch": epoch_id}
            for key, value in result_dict.items():
                if isinstance(value, list):
                    for idx, v in enumerate(value):
                        flat_metrics[f"{key}/{idx}"] = v
                else:
                    flat_metrics[key] = value
            wandb.log(flat_metrics)

    wandb.finish()

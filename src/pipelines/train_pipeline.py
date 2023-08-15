import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.data.datasets import AlfredDataset
from src.data.shemas import ConfigData
from src.models.MultyModalGPT2 import MultyModalGPT2
from src.utils import decode_from_logits
from src.pipelines.metrics import count_acc, count_cer, count_metrics


def train_batch(
    model: MultyModalGPT2,
    dataset: AlfredDataset,
    opt: AdamW,
    cfg: ConfigData
):
    # TODO:
    # - 8 is a max batch size. Set it to some const var
    loss_history = []
    model.train()

    train_data_size = None
    if cfg.train_cfg.data_samples_num == -1:
        train_data_size = len(dataset)
    else:
        train_data_size = min(len(dataset), cfg.train_cfg.data_samples_num)

    for i in tqdm(range(train_data_size), desc="TRAIN"):
        sample = dataset[i]
        if sample["images_features"] is not None:
            sample["images_features"] = sample["images_features"].to(
                cfg.device)

            data_size = min(8,  sample["images_features"].shape[0])
            sample_part = {
                k: v[:data_size] for k, v in sample.items()
            }
        else:
            sample_part = sample
        opt.zero_grad()
        out = model(sample_part)
        out["loss"].backward()
        opt.step()

        loss_history.append(out["loss"].item())
        torch.cuda.empty_cache()

    return {"loss": loss_history}


def eval_batch(
    model: MultyModalGPT2,
    dataset: AlfredDataset,
    cfg: ConfigData
):
    model.eval()

    loss_history = []
    acc_history = []
    cer_history = []

    eval_data_size = None
    if cfg.train_cfg.data_samples_num == -1:
        eval_data_size = len(dataset)
    else:
        eval_data_size = min(len(dataset), cfg.train_cfg.data_samples_num)

    for i in tqdm(range(eval_data_size), desc="EVAL "):
        sample = dataset[i]
        if sample["images_features"] is not None:
            sample["images_features"] = sample["images_features"].to(
                cfg.device)

            data_size = min(8,  sample["images_features"].shape[0])
            sample_part = {
                k: v[:data_size] for k, v in sample.items()
            }
        else:
            sample_part = sample
        with torch.no_grad():
            out = model(sample_part)

        predictions = decode_from_logits(
            model.mmtokenizer.tokenizer, out["logits"])
        metrics = count_metrics(predictions, sample_part["targets"])
        acc = metrics["accuracy"]
        cer = metrics["cer"]

        acc_history.append(acc)
        loss_history.append(out["loss"].item())
        cer_history.append(cer)

        torch.cuda.empty_cache()

    return {
        "loss": loss_history,
        "accuracy": acc_history,
        "cer": cer_history
    }


def train_loop(
    model: MultyModalGPT2,
    train_dataset: AlfredDataset,
    test_dataset: AlfredDataset,
    opt: AdamW,
    cfg: ConfigData
):
    train_loss = []
    test_loss = []

    train_acc = []
    test_acc = []

    train_cer = []
    test_cer = []

    for epoch in range(cfg.train_cfg.epoch_num):
        print(f"Epoch {epoch+1}/{cfg.train_cfg.epoch_num}")
        train_history = train_batch(model, train_dataset, opt, cfg)
        train_eval_history = eval_batch(model, train_dataset, cfg)
        test_eval_history = eval_batch(model, test_dataset, cfg)

        train_loss.append(np.mean(train_history["loss"]))
        train_acc.append(np.mean(train_eval_history["accuracy"]))
        train_cer.append(np.mean(train_eval_history["cer"]))

        test_loss.append(np.mean(test_eval_history["loss"]))
        test_acc.append(np.mean(test_eval_history["accuracy"]))
        test_cer.append(np.mean(test_eval_history["cer"]))

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_cer": train_cer,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_cer": test_cer
    }

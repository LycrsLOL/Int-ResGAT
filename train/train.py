import argparse
import logging
import math
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from train.loss import MultilabelFocalLoss
from train.models import HE_ResGATConv
from train.utils import create_dataloaders

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="The usage of `scatter(reduce='max')")


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        return local_rank, True
    return 0, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, is_distributed, print_freq=100):
    if is_distributed:
        loader.sampler.set_epoch(epoch)

    model.train()
    total_loss = 0.0
    total_steps = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y.float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        step_idx = batch_idx + 1
        if step_idx == 1 or step_idx % print_freq == 0 or step_idx == total_steps:
            if not is_distributed or dist.get_rank() == 0:
                logging.info(f"Epoch: [{epoch}] Step : [{step_idx}/{total_steps}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    if is_distributed:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / dist.get_world_size()

    return avg_loss


def _topk_hit_rate(y_true, y_score, k):
    k = min(k, y_score.shape[1])
    topk = np.argpartition(-y_score, kth=k - 1, axis=1)[:, :k]
    hits = [y_true[i, topk[i]].sum() > 0 for i in range(y_true.shape[0])]
    return float(np.mean(hits)) if hits else 0.0


def _safe_multilabel_auc(metric_fn, y_true, y_score):
    valid = [idx for idx in range(y_true.shape[1]) if len(np.unique(y_true[:, idx])) == 2]
    if not valid:
        return 0.0
    try:
        return float(metric_fn(y_true[:, valid], y_score[:, valid], average="macro"))
    except Exception:
        return 0.0


def _fmax(y_true, y_score):
    best = 0.0
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_score >= threshold).astype(np.int64)
        score = f1_score(y_true, y_pred, average="micro", zero_division=0)
        best = max(best, float(score))
    return best


def evaluate(model, loader, criterion, device, is_distributed, threshold=0.5, ensure_min_prediction=True):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    is_main_process = not is_distributed or (dist.is_initialized() and dist.get_rank() == 0)
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", disable=not is_main_process)
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.float())

            total_loss += loss.item() * batch.num_graphs
            all_logits.append(out)
            all_labels.append(batch.y.float())

    local_logits = torch.cat(all_logits)
    local_labels = torch.cat(all_labels)

    if is_distributed:
        loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / len(loader.dataset)
        np_logits = gather_tensor(local_logits).cpu().numpy()
        np_labels = gather_tensor(local_labels).cpu().numpy()
    else:
        global_avg_loss = total_loss / len(loader.dataset)
        np_logits = local_logits.cpu().numpy()
        np_labels = local_labels.cpu().numpy()

    np_labels = np_labels.astype(np.int64, copy=False)
    np_probs = 1.0 / (1.0 + np.exp(-np_logits))
    np_preds = (np_probs >= threshold).astype(np.int64)

    if ensure_min_prediction and np_preds.shape[1] > 0:
        empty_rows = np.where(np_preds.sum(axis=1) == 0)[0]
        if empty_rows.size:
            np_preds[empty_rows, np_probs[empty_rows].argmax(axis=1)] = 1

    metrics = {
        "loss": global_avg_loss,
        "threshold": threshold,
        "subset_acc": float(np.mean(np.all(np_preds == np_labels, axis=1))),
        "top1_acc": _topk_hit_rate(np_labels, np_probs, 1),
        "top5_acc": _topk_hit_rate(np_labels, np_probs, 5),
        "label_cardinality_true": float(np_labels.sum(axis=1).mean()),
        "label_cardinality_pred": float(np_preds.sum(axis=1).mean()),
        "precision_macro": precision_score(np_labels, np_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(np_labels, np_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(np_labels, np_preds, average="macro", zero_division=0),
        "precision_micro": precision_score(np_labels, np_preds, average="micro", zero_division=0),
        "recall_micro": recall_score(np_labels, np_preds, average="micro", zero_division=0),
        "f1_micro": f1_score(np_labels, np_preds, average="micro", zero_division=0),
        "precision_weighted": precision_score(np_labels, np_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(np_labels, np_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(np_labels, np_preds, average="weighted", zero_division=0),
        "fmax": _fmax(np_labels, np_probs),
    }
    metrics["auroc_macro"] = _safe_multilabel_auc(roc_auc_score, np_labels, np_probs)
    metrics["auprc_macro"] = _safe_multilabel_auc(average_precision_score, np_labels, np_probs)
    return metrics


def run_train(args):
    local_rank, is_distributed = setup_distributed()
    is_main_process = local_rank == 0

    try:
        config_path = getattr(args, "config", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if is_main_process:
            logging.info("========== Training Configuration ==========")
            train_config_log = config.get("train", {})
            for key, value in train_config_log.items():
                logging.info(f"  {key}: {value}")
            logging.info("============================================")

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if is_main_process:
            logging.info(f"Using device: {device} (Distributed: {is_distributed})")
            logging.info("Loading data...")

        train_loader, test_loader, num_classes, class_weights, label_names = create_dataloaders(
            config,
            distributed=is_distributed,
        )

        if is_main_process:
            world_size = dist.get_world_size() if is_distributed else 1
            train_samples = len(train_loader.dataset)
            test_samples = len(test_loader.dataset)
            batch_size_per_rank = config.get("train", {}).get("batch_size", 1)
            global_batch_size = batch_size_per_rank * world_size
            expected_train_steps = math.ceil(train_samples / global_batch_size)
            expected_test_steps = math.ceil(test_samples / global_batch_size)
            logging.info(
                f"Data loaded. Train steps per epoch (optimizer updates): {len(train_loader)} | "
                f"Expected: {expected_train_steps} | Train samples: {train_samples} | "
                f"Batch/GPU: {batch_size_per_rank} | World Size: {world_size} | Global Batch: {global_batch_size}"
            )
            logging.info(
                f"Data loaded. Test steps per epoch: {len(test_loader)} | "
                f"Expected: {expected_test_steps} | Test samples: {test_samples}"
            )

        train_config = config["train"]
        model = HE_ResGATConv(
            input_dim=train_config["input_dim"],
            hidden_dim=train_config["hidden_dim"],
            num_classes=num_classes,
            dropout=train_config["dropout"],
            num_layers=train_config["num_layers"],
            heads=train_config["heads"],
        ).to(device)

        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        alpha = train_config.get("focal_loss_alpha")
        pos_weight = None
        if isinstance(alpha, str) and alpha.lower() == "auto":
            alpha = None
            pos_weight = class_weights.to(device)
            if is_main_process:
                logging.info(
                    "Using automatic positive class weights for multi-label focal loss. "
                    f"Range: [{pos_weight.min():.4f}, {pos_weight.max():.4f}]"
                )

        criterion = MultilabelFocalLoss(
            alpha=alpha,
            gamma=train_config["focal_loss_gamma"],
            pos_weight=pos_weight,
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config.get("weight_decay", 0.0),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=train_config.get("lr_patience", 10),
        )

        epochs = train_config["epochs"]
        save_dir = config.get("save_dir") or train_config.get("save_dir")
        if not save_dir:
            raise ValueError("save_dir not found in config")
        print_freq = config.get("print_freq", 100)
        threshold = float(train_config.get("prediction_threshold", 0.5))
        ensure_min_prediction = bool(train_config.get("ensure_min_prediction", True))

        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        best_f1 = 0.0
        best_val_loss = float("inf")
        early_stopping_counter = 0
        start_epoch = 1

        if hasattr(args, "resume") and args.resume:
            if os.path.isfile(args.resume):
                if is_main_process:
                    logging.info(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint["epoch"] + 1
                best_f1 = checkpoint["best_f1"]
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                early_stopping_counter = checkpoint.get("early_stopping_counter", 0)

                if is_distributed:
                    model.module.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint["state_dict"])

                optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint and scheduler:
                    scheduler.load_state_dict(checkpoint["scheduler"])

                if is_main_process:
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            elif is_main_process:
                logging.info(f"=> no checkpoint found at '{args.resume}'")

        if is_main_process:
            logging.info("Start training...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                is_distributed,
                print_freq=print_freq,
            )
            test_metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
                is_distributed,
                threshold=threshold,
                ensure_min_prediction=ensure_min_prediction,
            )

            scheduler.step(test_metrics["loss"])

            if test_metrics["loss"] < best_val_loss:
                best_val_loss = test_metrics["loss"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= train_config.get("early_stopping_patience", 20):
                if epoch >= train_config.get("min_epochs", 0):
                    if is_main_process:
                        logging.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(min_epochs={train_config.get('min_epochs', 0)})."
                        )
                    break
                elif is_main_process and epoch % print_freq == 0:
                    logging.info(
                        f"Early stopping condition met (counter={early_stopping_counter}), "
                        f"but waiting for min_epochs ({train_config.get('min_epochs', 0)}). "
                        f"Current epoch: {epoch}"
                    )

            if is_main_process:
                log_str = (
                    f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | "
                    f"Test Loss: {test_metrics['loss']:.4f} | "
                    f"Top-1: {test_metrics['top1_acc']:.4f} | "
                    f"Top-5: {test_metrics['top5_acc']:.4f} | "
                    f"SubsetAcc: {test_metrics['subset_acc']:.4f} | "
                    f"F1(Ma): {test_metrics['f1_macro']:.4f} | "
                    f"F1(Mi): {test_metrics['f1_micro']:.4f} | "
                    f"Prec(Ma): {test_metrics['precision_macro']:.4f} | "
                    f"Rec(Ma): {test_metrics['recall_macro']:.4f} | "
                    f"Fmax: {test_metrics['fmax']:.4f}"
                )
                if test_metrics["auroc_macro"] > 0:
                    log_str += f" | AUROC(Ma): {test_metrics['auroc_macro']:.4f}"
                if test_metrics["auprc_macro"] > 0:
                    log_str += f" | AUPRC(Ma): {test_metrics['auprc_macro']:.4f}"
                logging.info(log_str)

                current_f1 = test_metrics["f1_macro"]
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    save_subdir = os.path.join(save_dir, "best_model")
                    os.makedirs(save_subdir, exist_ok=True)
                    save_path = os.path.join(save_subdir, f"train_{timestamp}.pth")
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), save_path)
                    logging.info(f"--> Best model saved with Macro-F1: {best_f1:.4f} to {save_path}")

                checkpoint_state = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict() if is_distributed else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "best_f1": best_f1,
                    "best_val_loss": best_val_loss,
                    "early_stopping_counter": early_stopping_counter,
                    "label_names": label_names,
                    "prediction_threshold": threshold,
                }
                latest_checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
                torch.save(checkpoint_state, latest_checkpoint_path)

        if is_main_process:
            logging.info("Training completed.")

    except KeyboardInterrupt:
        if is_main_process:
            logging.warning("Training interrupted by user. Cleaning up...")
    except Exception as e:
        if is_main_process:
            logging.error(f"An error occurred: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Int-ResGAT")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default="", help="Path to latest checkpoint (default: none)")
    args = parser.parse_args()

    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")],
    )
    logging.info(f" >> Log File      : {log_file}")

    run_train(args)


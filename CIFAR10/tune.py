# tune.py - 手动多组参数对比版
import argparse
import datetime
import itertools
import json
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from models import build_model
from utils import set_seed, save_env_snapshot, calculate_topk_accuracy

# 默认手动测试组合 lr  batchsize  weightdecay  optimize:SGD/AdamW
PARAM_GRID = [
    {"trial_name": "exp_lr0.01_bs16",  "lr": 0.01,  "batch_size": 16, "weight_decay": 5e-4, "optimizer": "SGD"},
    {"trial_name": "exp_lr0.01_bs32",  "lr": 0.01,  "batch_size": 32, "weight_decay": 5e-4, "optimizer": "SGD"},
    {"trial_name": "exp_lr0.001_bs16", "lr": 0.001, "batch_size": 16, "weight_decay": 5e-4, "optimizer": "SGD"},
    {"trial_name": "exp_lr0.001_bs32", "lr": 0.001, "batch_size": 32, "weight_decay": 5e-4, "optimizer": "SGD"},
]

def run_single_trial(trial_id, params, args, output_dir):
    set_seed(args.seed)

    lr = params["lr"]
    batch_size = params["batch_size"]
    weight_decay = params.get("weight_decay", 5e-4)
    optimizer_name = params.get("optimizer", "SGD")
    trial_name = params.get("trial_name", f"trial_{trial_id}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 1. 先加载数据
    train_loader, val_loader, _, num_classes = get_dataloaders(dataset_name=args.dataset, batch_size=batch_size)

    # 2. 实例化 model 变量（修复 NameError 的关键）
    model = build_model(model_name=args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) if optimizer_name == "SGD" \
        else optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard", trial_name))
    trial_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    logging.info(f"\n>>>> [Trial {trial_id}/{args.total_trials}] 开始评估配置: {params} <<<<")

    for epoch in range(1, args.tune_epochs + 1):
        # 训练过程
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            top1, _ = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            train_correct += (top1 / 100.0) * targets.size(0)
            train_total += targets.size(0)

        train_loss /= train_total

        # 验证集评估
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                top1, _ = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
                val_correct += (top1 / 100.0) * targets.size(0)
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial_history["epoch"].append(epoch)
        trial_history["train_loss"].append(train_loss)
        trial_history["val_loss"].append(val_loss)
        trial_history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        logging.info(
            f"Trial {trial_id} | Epoch [{epoch:02d}/{args.tune_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    writer.close()
    return best_val_acc, trial_history


def main():
    parser = argparse.ArgumentParser(description="手动多组超参数试验对比 (Grid Search)")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="alexnet")
    parser.add_argument("--tune-epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # 注册 --lr-list 和 --bs-list 参数
    parser.add_argument("--lr-list", nargs="+", type=float, default=[], help="例: --lr-list 0.01 0.001")
    parser.add_argument("--bs-list", nargs="+", type=int, default=[], help="例: --bs-list 16 32")

    args = parser.parse_args()

    if args.lr_list and args.bs_list:
        trials_to_run = []
        for idx, (lr, bs) in enumerate(itertools.product(args.lr_list, args.bs_list), 1):
            trials_to_run.append({
                "trial_name": f"cli_lr{lr}_bs{bs}",
                "lr": lr,
                "batch_size": bs,
                "weight_decay": 5e-4,
                "optimizer": "SGD"
            })
    else:
        trials_to_run = PARAM_GRID

    args.total_trials = len(trials_to_run)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"manual_tune_{timestamp}_{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(output_dir, "tune.log")),
        logging.StreamHandler()
    ])

    save_env_snapshot(os.path.join(output_dir, "env_snapshot.json"))

    logging.info(f"==> 开始执行手动超参数调优试验 (共 {args.total_trials} 组配置)")
    logging.info(f"==> 独立输出目录: {output_dir}")

    summary_records = []
    all_trial_metrics = {}
    best_overall_acc = -1.0
    best_params = None

    for idx, params in enumerate(trials_to_run, 1):
        trial_name = params.get("trial_name", f"trial_{idx}")
        best_val_acc, trial_history = run_single_trial(idx, params, args, output_dir)

        all_trial_metrics[trial_name] = trial_history

        record = {
            "trial_id": idx,
            "trial_name": trial_name,
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "weight_decay": params.get("weight_decay", 5e-4),
            "optimizer": params.get("optimizer", "SGD"),
            "best_val_acc": best_val_acc
        }
        summary_records.append(record)

        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_params = params

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(output_dir, "hpo_trials.csv"), index=False)

    with open(os.path.join(output_dir, "trial_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_trial_metrics, f, indent=4)

    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    logging.info("\n==================== 手动调优实验汇总对比 ====================")
    logging.info("\n" + df_summary.to_string(index=False))
    logging.info("==============================================================")
    logging.info(f"--> 最优配置已写出至: {best_params_path}")
    logging.info(f"--> 最佳验证集 Acc: {best_overall_acc:.2f}% (配置: {best_params})")

if __name__ == "__main__":
    main()
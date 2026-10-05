# train.py
import argparse
import datetime
import json
import logging
import os
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from dataset import get_dataloaders
from models import build_model
from utils import set_seed, save_env_snapshot, calculate_topk_accuracy, EMA, EarlyStopping
from visualize import plot_learning_curves, plot_confusion_matrix, save_training_data

def main():
    parser = argparse.ArgumentParser(description="正式训练框架 (带测试集完整准确率评估)")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="alexnet")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--best-params", type=str, default="", help="读取 best_params.json 覆盖参数")
    parser.add_argument("--resume", type=str, default="", help="指定 last.pth 文件路径恢复训练")

    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    # 1. 固定随机种子
    set_seed(args.seed)

    # 2. 读取 HPO 配置文件
    if args.best_params and os.path.exists(args.best_params):
        with open(args.best_params, "r", encoding="utf-8") as f:
            hpo_cfg = json.load(f)
            args.lr = hpo_cfg.get("lr", args.lr)
            args.weight_decay = hpo_cfg.get("weight_decay", args.weight_decay)
            args.batch_size = hpo_cfg.get("batch_size", args.batch_size)
            print(f"--> [HPO Loaded] 读取配置文件成功: LR={args.lr}, BS={args.batch_size}")

    # 3. 创建输出目录与 Log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"train_{timestamp}_{args.model}")
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(output_dir, "train.log")),
        logging.StreamHandler()
    ])

    save_env_snapshot(os.path.join(output_dir, "env_snapshot.json"))
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(dataset_name=args.dataset, batch_size=args.batch_size)

    model = build_model(model_name=args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    ema = EMA(model) if args.use_ema else None
    early_stopping = EarlyStopping(patience=args.patience)

    start_epoch = 1
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 断点续训恢复机制
    if args.resume and os.path.exists(args.resume):
        logging.info(f"--> [Resume] 从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', history)
        logging.info(f"--> [Resume] 成功！从 Epoch {start_epoch} 继续。")

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 4. 训练主循环
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}", leave=False)
        for images, targets in pbar:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema:
                ema.update()

            train_loss += loss.item() * images.size(0)
            top1, _ = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            train_correct += (top1 / 100.0) * targets.size(0)
            train_total += targets.size(0)

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # 验证阶段
        if ema:
            ema.apply_shadow()

        model.eval()
        val_loss, val_correct_top1, val_correct_top5, val_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
                val_correct_top1 += (top1 / 100.0) * targets.size(0)
                val_correct_top5 += (top5 / 100.0) * targets.size(0)
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc_top1 = 100.0 * val_correct_top1 / val_total
        val_acc_top5 = 100.0 * val_correct_top5 / val_total

        if ema:
            ema.restore()

        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc_top1)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val_top1", val_acc_top1, epoch)
        writer.add_scalar("Accuracy/val_top5", val_acc_top5, epoch)

        logging.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Top-1: {val_acc_top1:.2f}% Top-5: {val_acc_top5:.2f}%"
        )

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': max(best_val_acc, val_acc_top1),
            'history': history,
        }
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, "last.pth"))

        if val_acc_top1 > best_val_acc:
            best_val_acc = val_acc_top1
            torch.save(checkpoint_state, os.path.join(checkpoint_dir, "best.pth"))
            logging.info(f"--> [Best Model] 最优权重已保存！(Val Top-1 Acc: {val_acc_top1:.2f}%)")

        early_stopping(val_acc_top1)
        if early_stopping.early_stop:
            logging.info("--> [Early Stop] 提前终止训练！")
            break

    # 绘制训练曲线与保存 JSON 数据
    plot_learning_curves(history, save_path=os.path.join(output_dir, "plots", "learning_curves.png"))

    # =========================================================================
    # 5. 独立测试集 (Test Set) 评估输出 (修复核心点)
    # =========================================================================
    logging.info("\n==============================================================")
    logging.info("==> 训练结束，加载最佳模型 (best.pth) 进行独立测试集 (Test Set) 评估...")
    
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])

    if ema:
        ema.apply_shadow()

    model.eval()
    test_loss, test_correct_top1, test_correct_top5, test_total = 0.0, 0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="Testing", leave=False)
        for images, targets in pbar_test:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * images.size(0)

            # 计算 Top-1 和 Top-5 准确率
            top1, top5 = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
            test_correct_top1 += (top1 / 100.0) * targets.size(0)
            test_correct_top5 += (top5 / 100.0) * targets.size(0)
            test_total += targets.size(0)

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_loss /= test_total
    test_acc_top1 = 100.0 * test_correct_top1 / test_total
    test_acc_top5 = 100.0 * test_correct_top5 / test_total

    # 控制台高亮+文件日志输出测试结果
    logging.info("======================== 测试集评估结果 ========================")
    logging.info(f"--> Test Loss:     {test_loss:.4f}")
    logging.info(f"--> Test Top-1 Acc: {test_acc_top1:.2f}%")
    logging.info(f"--> Test Top-5 Acc: {test_acc_top5:.2f}%")
    logging.info("==============================================================")

    # 将测试集结果追加保存至 JSON 数据文件
    history['final_test_loss'] = test_loss
    history['final_test_top1_acc'] = test_acc_top1
    history['final_test_top5_acc'] = test_acc_top5
    save_training_data(history, save_path=os.path.join(output_dir, "plots", "training_history.json"))

    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    classes = [str(i) for i in range(num_classes)]
    plot_confusion_matrix(cm, classes, save_path=os.path.join(output_dir, "plots", "confusion_matrix.png"))
    logging.info("--> 混淆矩阵 (confusion_matrix.png) 生成完毕！\n")

    writer.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("训练遭遇致命异常：")
        logging.error(traceback.format_exc())
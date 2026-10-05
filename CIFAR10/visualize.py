# visualize.py  
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(history, save_path):
    """绘制训练/验证 Loss & Top-1 Acc 曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    plt.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', markersize=3)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Train Top-1 Acc', markersize=3)
    plt.plot(epochs, history['val_acc'], 'g-^', label='Val Top-1 Acc', markersize=3)
    plt.title('Training & Validation Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path):
    """【P1】纯手绘混淆矩阵热力图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_optuna_visualizations(study, save_dir):
    """【P1】利用 Optuna 官方 matplotlib 模块绘制参数重要性与平行坐标图"""
    import optuna.visualization.matplotlib as ovm
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 参数重要性图
        ax = ovm.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "hpo_param_importances.png"), dpi=300)
        plt.close()

        # 平行坐标图
        ax = ovm.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "hpo_parallel_coordinate.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"--> [Warning] 绘制 Optuna 图像时跳过 (试炼数量可能不足): {e}")
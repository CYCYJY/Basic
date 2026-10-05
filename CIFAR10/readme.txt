watch -n 1 -d nvidia-smi
功能使用指南（断点续训、HPO 无缝衔接等）
在代码重构完成后，以下是你在服务器上使用新特性的操作指令：


使用说明与命令行示例


cd basic/CIFAR10
1. 使用方式 1：直接在代码中修改 PARAM_GRID（最直观）
打开 tune.py，修改文件头部的 PARAM_GRID 列表，加入你想测试的组合：

Python
PARAM_GRID = [
    {"trial_name": "vgg_lr0.01_bs64",  "lr": 0.01,  "batch_size": 64},
    {"trial_name": "vgg_lr0.01_bs128", "lr": 0.01,  "batch_size": 128},
    {"trial_name": "vgg_lr0.005_bs64", "lr": 0.005, "batch_size": 64},
]
运行命令：

Bash
python3 tune.py --dataset cifar10 --model vgg16 --tune-epochs 5 --gpu 0
2. 使用方式 2：在终端命令行动态传入列表（最快捷）
如果你不想改代码，可直接在命令行传入要对比的 lr 和 batch_size 列表，脚本会自动做交叉组合（笛卡尔积）：

Bash
# 传入 2 个 lr 和 2 个 bs，会自动运行 2x2 = 4 组对照试验
python3 tune.py   --dataset cifar10   --model alexnet   --lr-list 0.01 0.001   --bs-list 16 32   --tune-epochs 5   --gpu 0
3. 调优完成后无缝衔接正式训练
试验完成后，终端打印出的汇总表以及输出目录中的 best_params.json 会记录表现最好的一组参数，直接供 train.py 使用：

Bash
python3 train.py   --dataset cifar10   --model alexnet   --best-params outputsmanual_tune_20261001_134159_alexnet/best_params.json   --epochs 40   --gpu 0
python3 train.py   --dataset cifar10   --model alexnet   --best-params outputsmanual_tune_20261001_134159_alexnet/best_params.json   --epochs 40   --gpu 0
3. 断点续训（Resume Training）如何无缝衔接？
如果训练中途遭遇服务器断电、显存爆满崩溃或人为中断，可以通过 --resume 指向目标运行目录中的 last.pth 恢复训练：

Bash
# 假设训练在第 8 Epoch 中断，运行下面命令即可无缝接着第 9 Epoch 运行，所有的 LR Schedule/Optimizer 状态与种子依然完美对齐：
python3 train.py \
  --dataset cifar10 \
  --model vgg16 \
  --resume outputs/train_20260925_180000_vgg16/checkpoints/last.pth \
  --epochs 20 \
  --gpu 0
4. 实时查看 TensorBoard
在命令行终端输入以下命令查看交互式曲线：

Bash
tensorboard --logdir=outputs/train_20261001_154612_alexnet/tensorboard


四、 工作会话存档快照 (Context Snapshot)
为保证下次 AI 对话无缝衔接，以下是当前最新的项目与系统存盘信息：

# AI 对话上下文无缝衔接存档 (Updated Context Snapshot)
系统与环境:
  Host: zhanglei (Ubuntu 18.04, RTX 3090 x 3)
  User: yangjunyi
  Working_Directory: /data1/users/yangjunyi/basic/my_cv_framework
  Python/PyTorch: Python 3.6.9 / PyTorch 1.7.1+cu110

本次改动 (tune.py 模式重构):
  1. tune.py: 移除了 Optuna 随机采样库，改造为手动多组超参数试验模式 (Manual Sweep / Grid Search)。
  2. 配置接口: 支持通过代码 PARAM_GRID 列表或命令行 --lr-list / --bs-list 两种方式设置试炼组合。
  3. 数据与展示: 每次运行生成独立的 outputs/manual_tune_YYYYMMDD_HHMMSS_vgg16 文件夹，保存对比表格 hpo_trials.csv、逐 Epoch 详细指标 trial_metrics.json 与最优参数 best_params.json。

后续计划/待办事项 (Next Steps):
  - 任务 1: 编写 models/alexnet.py，实现 AlexNet 并集成至框架。
  - 任务 2: 编写 models/googlenet.py (Inception v1 结构)。
  - 任务 3: 编写 models/resnet.py (BasicBlock / Bottleneck 结构)。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器配置信息收集脚本
用法: python3 sysinfo.py
"""

import sys
import os
import platform
import subprocess
import traceback

def run_cmd(cmd):
    """执行 shell 命令并返回输出（合并 stderr）"""
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        if isinstance(out, bytes):
            out = out.decode('utf-8', errors='replace')
        return out.strip()
    except subprocess.CalledProcessError as e:
        out = e.output
        if isinstance(out, bytes):
            out = out.decode('utf-8', errors='replace')
        return out.strip()
    except Exception as e:
        return "执行失败: " + str(e)

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ==================== 1. 系统信息 ====================
section("1. 系统信息")
print("主机名:", platform.node())
print("系统:", platform.system(), platform.release())
print("版本:", platform.version())
print("架构:", platform.machine())
print("处理器:", platform.processor())
print("\n/etc/os-release:")
print(run_cmd("cat /etc/os-release 2>/dev/null"))
print("\nuname -a:")
print(run_cmd("uname -a"))
print("\nlscpu (前40行):")
print(run_cmd("lscpu 2>/dev/null | head -40"))
print("\n内存:")
print(run_cmd("free -h 2>/dev/null"))
print("\n磁盘:")
print(run_cmd("df -h 2>/dev/null | head -20"))

# ==================== 2. Python 环境 ====================
section("2. Python 环境")
print("当前脚本 Python 版本:", sys.version)
print("当前脚本可执行文件:", sys.executable)
print("\nsys.path:")
for p in sys.path:
    print("  ", p)

print("\nwhich python:", run_cmd("which python"))
print("python --version:", run_cmd("python --version 2>&1"))
print("which python3:", run_cmd("which python3"))
print("python3 --version:", run_cmd("python3 --version 2>&1"))
print("which pip:", run_cmd("which pip"))
print("pip -V:", run_cmd("pip -V 2>&1"))
print("python3 -m pip -V:", run_cmd("python3 -m pip -V 2>&1"))

# ==================== 3. pip 关键包信息 ====================
section("3. pip 关键包信息")
py = sys.executable
key_packages = [
    "torch", "torchvision", "torchaudio",
    "numpy", "tensorflow", "tensorflow-gpu",
    "cupy", "pynvml", "nvidia-ml-py",
    "scipy", "pandas", "scikit-learn", "matplotlib"
]
for pkg in key_packages:
    print(f"\n--- {pkg} (通过 {py} -m pip show) ---")
    print(run_cmd(f'"{py}" -m pip show {pkg} 2>&1'))

print("\n--- 默认 pip show torch ---")
print(run_cmd("pip show torch 2>&1"))

print("\n--- pip list 中 torch/cuda/nvidia/tensorflow 相关 ---")
print(run_cmd("pip list 2>/dev/null | grep -i -E 'torch|cuda|nvidia|cupy|tensorflow'"))

# ==================== 4. PyTorch 详细信息 ====================
section("4. PyTorch 详细信息")
try:
    import torch
    print("当前 Python 解释器 import torch 成功")
    print("torch.__version__:", torch.__version__)
    print("torch.__file__:", torch.__file__)
    print("torch.version.cuda:", getattr(torch.version, 'cuda', None))
    if hasattr(torch.backends, 'cudnn'):
        print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("torch.cuda.current_device():", torch.cuda.current_device())
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        print("torch.cuda.get_device_capability(0):", torch.cuda.get_device_capability(0))
        print("torch.cuda.get_device_properties(0):", torch.cuda.get_device_properties(0))
    else:
        print("CUDA 不可用")
except Exception as e:
    print("当前 Python 解释器 import torch 失败:", repr(e))
    traceback.print_exc()
    print("\n尝试用 python3 子进程导入 torch:")
    print(run_cmd('python3 -c "import torch; print(torch.__version__); print(torch.__file__); print(torch.version.cuda); print(torch.cuda.is_available())" 2>&1'))

# ==================== 5. GPU / CUDA / 驱动 ====================
section("5. GPU / CUDA / 驱动")
print("nvidia-smi:")
print(run_cmd("nvidia-smi 2>&1"))
print("\nnvcc --version:")
print(run_cmd("nvcc --version 2>&1"))
print("\nCUDA 相关环境变量:")
for var in ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH", "CUDA_VISIBLE_DEVICES"]:
    print(f"{var}={os.environ.get(var, '')}")

# ==================== 6. conda 信息 ====================
section("6. conda 信息")
print("which conda:", run_cmd("which conda 2>&1"))
print("conda --version:", run_cmd("conda --version 2>&1"))
print("conda env list:", run_cmd("conda env list 2>&1"))
print("常见 conda 安装路径:")
print(run_cmd("ls -d /opt/miniconda3 /opt/anaconda3 ~/miniconda3 ~/anaconda3 2>/dev/null"))

# ==================== 7. 其他 ====================
section("7. 其他")
print("环境变量 PATH:", os.environ.get("PATH", ""))
print("当前工作目录:", os.getcwd())
print("当前用户:", run_cmd("whoami"))
print("用户 home:", os.path.expanduser("~"))

print("\n" + "=" * 70)
print("信息收集完成。请将以上输出复制给 AI。")
print("=" * 70)
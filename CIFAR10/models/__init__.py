# models/__init__.py
from .vgg import VGG
from .alexnet import AlexNet  # 引入手写的 AlexNet

def build_model(model_name='vgg16', num_classes=10):
    """
    模型工厂函数：根据 model_name 动态构建模型
    """
    model_name = model_name.lower()
    if 'vgg' in model_name:
        return VGG(cfg_name=model_name, num_classes=num_classes, batch_norm=True)
    elif 'alexnet' in model_name:
        return AlexNet(num_classes=num_classes)
    # 留出未来任务扩展:
    # elif 'googlenet' in model_name:
    #     return GoogLeNet(num_classes=num_classes)
    # elif 'resnet' in model_name:
    #     return ResNet(model_name, num_classes=num_classes)
    else:
        raise ValueError(f"未注册的模型名称: {model_name}")
    

#     原因 2：修改了文件但未保存，或存在文件路径/PyTorch 缓存覆盖
# 请检查你在终端运行命令时的当前路径是 /data1/users/yangjunyi/basic/CIFAR10：

# 确保你修改并保存的 models/__init__.py 是 /data1/users/yangjunyi/basic/CIFAR10/models/__init__.py。

# 检查目录下是否存在 PyTorch 或 Python 编译缓存文件（如 models/__pycache__/），它可能导致 Python 仍然加载修改前的旧字节码。

# ResNet50 Flower Classification

基于 MindSpore 的 ResNet50 花卉图像分类项目，使用迁移学习对 5 种花卉（雏菊、蒲公英、玫瑰、向日葵、郁金香）进行分类。

## 项目目录结构

```
ResNet50_Flower_Classification/    # 项目根目录
│
├── flowers/                       # 数据集根目录
│   ├── flower_photos_train/       # 训练集目录
│   │   ├── daisy/                 # 雏菊图片
│   │   ├── dandelion/             # 蒲公英图片
│   │   ├── roses/                 # 玫瑰图片
│   │   ├── sunflowers/            # 向日葵图片
│   │   └── tulips/                # 郁金香图片
│   │
│   └── flower_photos_test/        # 测试集目录
│       ├── daisy/                 # 雏菊图片
│       ├── dandelion/             # 蒲公英图片
│       ├── roses/                 # 玫瑰图片
│       ├── sunflowers/            # 向日葵图片
│       └── tulips/                # 郁金香图片
│
├── model_resnet/                  # 预训练模型及训练输出目录
│   └── resnet50_ascend_v170_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt
│
├── main.py                        # 主程序文件
├── requirements.txt               # 项目依赖
└── README.md
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据准备

### 1. 下载花卉数据集

实验使用的花卉图像数据集通过以下链接下载：

- **训练集**：<https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_train.zip>
- **测试集**：<https://ascend-professional-construction-dataset.obs.myhuaweicloud.com/deep-learning/flower_photos_test.zip>

下载并解压后，将图片分别放入对应目录下：

| 数据集 | 解压后目录 | 目标路径 |
|--------|-----------|----------|
| 训练集 | `flower_photos_train/` | `flowers/flower_photos_train/` |
| 测试集 | `flower_photos_test/` | `flowers/flower_photos_test/` |

每个分类下应包含对应花卉的 `.jpg` 图片文件。

### 2. 下载预训练模型

本项目使用官方在 ImageNet 上训练的 ResNet50 预训练模型权重文件：

- **预训练权重**：<https://download.mindspore.cn/models/r1.7/resnet50_ascend_v170_imagenet2012_official_cv_top1acc76.97_top5acc93.44.ckpt>

下载后放入 `model_resnet/` 目录下。

## 运行

```bash
python main.py
```

## 技术栈

- **框架**：MindSpore 2.x
- **模型**：ResNet50（迁移学习，修改全连接层适配 5 分类任务）
- **数据集**：花卉图像 5 分类（3,616 张训练图片）
- **输入尺寸**：224 × 224
- **优化器**：Momentum
- **损失函数**：SoftmaxCrossEntropyWithLogits

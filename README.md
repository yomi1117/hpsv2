# HPS v2 (简化版)

**注意：本 README 已被大幅简化，删除了图片和详细内容，仅保留基础信息。**

## 简介
HPS v2 是一个用于评测文本生成图像模型的工具和基准。我们拉下源仓库，希望可以为不同的reward model做匹配，可以eval benchmark接上所有的模型。

## 安装
```bash
pip install hpsv2
```
或本地安装：
```bash
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e .
```

## 快速开始
```python
import hpsv2
result = hpsv2.score(["img1.jpg", "img2.jpg"], "<prompt>", hps_version="v2.1")
```

## 评测自定义模型
```python
import hpsv2
hpsv2.evaluate("<images_path>", hps_version="v2.1")
```

---
如需详细文档和数据集，请访问原项目主页或 Hugging Face。

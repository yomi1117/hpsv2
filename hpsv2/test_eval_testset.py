import os
from hpsv2.src.training.data import RankingDataset, collate_rank
from torch.utils.data import DataLoader
from tqdm import tqdm
import ImageReward as RM
from hpsv2.src.training.train import inversion_score


def simple_collate(batch):
    return batch

# 配置参数
data_path = "/data/yangyuanming/dataset_temp/HPDv2"  # test.json 路径
image_folder = "/data/yangyuanming/dataset_temp/HPDv2/test"  # 图片文件夹路径
batch_size = 20

# 只加载数据集，不做任何模型相关操作
from hpsv2.src.open_clip import get_tokenizer
model_name = "ViT-H-14"
# tokenizer = get_tokenizer(model_name)
tokenizer = None


path = "/mnt/jfs/yym/ckpt/ImageReward/ImageReward.pt"
model = RM.load(path)
# 这里不传预处理函数
# RankingDataset(meta_file, image_folder, preprocess, tokenizer)
preprocess = lambda x: x
dataset = RankingDataset(os.path.join(data_path, 'test.json'), image_folder, preprocess, tokenizer)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_rank)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=simple_collate)

score = 0
for batch_idx, batch in enumerate(tqdm(dataloader)):
    # print(len(dataset))
    for sample in batch:
        # print(sample)
        # 遍历sample元素
        images, label, prompt = sample
        # print(f"图片列表: {type(images)}, 长度: {len(images)}")
        print(f"label: {type(label)}, 值: {label}")  
        print(f"prompt: {type(prompt)}, 值: {prompt}")

        # 使用模型进行推理
        ranking, rewards = model.inference_rank(prompt, images)
        print(ranking, rewards)
        score += inversion_score(ranking, label)
    print(f"score: {score/len(dataset)}")
        # images, num_images, labels, texts = sample
        # print(f"图片对象: {images}, 文本: {texts}")
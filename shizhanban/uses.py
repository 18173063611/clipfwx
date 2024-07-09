import os
import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from models import Mynet  # 假设你的多模态模型在models.py中定义
from Config import Config  # 假设你的配置文件在Config.py中定义

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图像模型和文本模型
config = Config()
image_model_path = 'model/S_minirbt-h256_resnet18.ckpt'  # 图像模型文件路径

# 初始化模型
model = Mynet(config)
model.load_state_dict(torch.load(image_model_path, map_location=device))
model.eval()
model.to(device)

# 加载文本模型和tokenizer
text_tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)  # 替换为你的BERT模型路径

# 图像预处理函数
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    return image

# 文本预处理函数
def preprocess_text(text):
    inputs = text_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    return inputs

# 批量预测函数
def batch_predict(image_paths, texts):
    results = []
    for image_path, text in zip(image_paths, texts):
        image_input = preprocess_image(image_path)
        text_input = preprocess_text(text)
        
        # 进行推断
        with torch.no_grad():
            # 提取图像特征和文本特征
            img_features, logits = model((image_input, text_input['input_ids'], text_input['attention_mask']))
            
            # 计算分类概率
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # 假设类别标签为['真实新闻', '虚假新闻']
            labels = ['真实新闻', '虚假新闻']
            predicted_label = labels[torch.argmax(logits).item()]
            
            results.append({
                "image_path": image_path,
                "text": text,
                "predicted_label": predicted_label,
                "probabilities": probs
            })
    return results

# 读取文件
image_file_path = 'images.txt'
text_file_path = 'texts.txt'

with open(image_file_path, 'r') as f:
    image_paths = [line.strip() for line in f.readlines()]

with open(text_file_path, 'r') as f:
    texts = [line.strip() for line in f.readlines()]

# 确保图像路径和文本数量一致
assert len(image_paths) == len(texts), "图像文件数量和文本数量不一致"

# 批量预测
results = batch_predict(image_paths, texts)

# 打印结果
for result in results:
    print(f"图像路径: {result['image_path']}")
    print(f"文本: {result['text']}")
    print(f"预测标签: {result['predicted_label']}")
    print(f"预测概率: {result['probabilities']}")
    print("---------")

from sentence_transformers import SentenceTransformer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 1. 指定模型名称
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

# 2. 加载模型（第一次运行会下载，大约 400MB-500MB）
# 如果你在国内下载慢，可以先设置环境变量：os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model = SentenceTransformer(model_name)

print("模型加载成功！")

# 3. 如果你想把模型下载到本地文件夹，方便以后离线使用：
model.save('./my_local_model') 
import numpy as np
from sentence_transformers import SentenceTransformer
import random
from sklearn.neural_network import MLPClassifier


mlp_save_path = 'model/MLP/command_classifier.pkl'

print('正在加载语言模型...')
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print('语言模型加载完成')

dataset = []
with open('dataset.txt', encoding='utf-8') as fp:
    for line in fp:
        l = line.strip().split(' | ')
        text = l[0]
        label = l[1]
        dataset.append({"text": text, "label": label})


random.seed(42)
random.shuffle(dataset)

train_dataset = dataset[:int(len(dataset) * 0.7)]
test_dataset = dataset[int(len(dataset) * 0.7):]

X_train = []
Y_train = []
for data in train_dataset:
    text = data["text"]
    label = data["label"]
    embedding = model.encode(text)
    X_train.append(embedding)
    Y_train.append(label)

X_test = []
Y_test = []
for data in test_dataset:
    text = data["text"]
    label = data["label"]
    embedding = model.encode(text)
    X_test.append(embedding)
    Y_test.append(label)

X = np.concatenate([X_train, X_test], axis=0)
Y = np.concatenate([Y_train, Y_test], axis=0)

print(f'成功导入{len(X)}条数据，自动划分训练集和测试集')
print(f'训练集：{len(X_train)}条，测试集：{len(X_test)}条')

print('\n正在训练语音指令分类模型...')
mlp = MLPClassifier(max_iter=2000, hidden_layer_sizes=(256, 128, 64))
mlp.fit(X_train, Y_train)
print('分类模型训练完成\n')

correct_count = 0
for x, y in zip(X_train, Y_train):
    pred = mlp.predict([x])
    if pred[0] == y:
        correct_count += 1
print(f"训练集准确度: {correct_count / len(X_train) * 100:.2f}%")

correct_count = 0
for x, y in zip(X_test, Y_test):
    pred = mlp.predict([x])
    if pred[0] == y:
        correct_count += 1
print(f"测试集准确度: {correct_count / len(X_test) * 100:.2f}%")

import joblib
joblib.dump(mlp, mlp_save_path)
print(f'分类模型已保存至：{mlp_save_path}')

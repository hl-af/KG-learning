"""
File: bert_test3.py
Author: xieyuyan
Date: 2024-06-29

Description:
集成test和train
通过RUN_TYPE调整训练和test
    RUN_TYPE = train
    RUN_TYPE = test
Inputs:
- bert-base-uncased

Outputs:
outputs bert的结构化输出

"""
# 假设你已经下载了WN18RR数据集并解压缩到适当的路径下
# bert模型测试
# 参考: https://zhuanlan.zhihu.com/p/605020970
import os
# 可选：进行数据预处理、处理成模型需要的输入格式
# 例如：将实体和关系映射成BERT的输入ID，或者使用预训练的BERT模型来获得实体和关系的表示
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

RUN_TYPE = 'test'

# 设置一些超参数
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
root_path = os.path.dirname(__file__)

# 读取数据集
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split('\t') for line in lines]
    return data


class BERT_KG(nn.Module):

    def __init__(self, num_relations, bert_model_name=os.path.join(root_path, 'checkpoints','bert-base-uncased')):
        super(BERT_KG, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size, num_relations)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 可以根据需要选择不同的输出

        logits = self.relation_classifier(pooled_output)
        return logits



def string_to_ascii_tensor(s):
    ascii_values = [ord(c) for c in s]  # 将每个字符转换为ASCII值
    tensor = torch.tensor(ascii_values, dtype=torch.long)  # 创建一个tensor
    return tensor

# 准备数据
def prepare_data(data,tokenizer):
    input_ids = []
    attention_masks = []
    labels = []

    for triple in data:
        # 处理三元组，获取实体和关系的BERT输入ID
        # 根据需要修改这部分的逻辑

        # 示例：假设三元组格式是（head_entity, relation, tail_entity）
        head_entity, relation, tail_entity = triple

        encoded_dict = tokenizer.encode_plus(
            head_entity, tail_entity,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(relation)  # 根据任务设置正确的标签（关系）

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_set=list(set(labels))
    labels_inx_lst=[labels_set.index(i) for i in labels]
    labels = torch.tensor(labels_inx_lst)

    return TensorDataset(input_ids, attention_masks, labels)



def train(model, optimizer, train_loader, num_epochs):

    predictions = []
    true_labels = []

    model.train()
    epoch=0
    # for epoch in range(num_epochs):
    # 不计算梯度
    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            #
            # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

            logits = outputs
            predictions.extend(logits.argmax(dim=-1).tolist())
            true_labels.extend(labels.tolist())
            print(1)
            torch.save(model.state_dict(), 'bert.model')


# 开始训练
# train(model, optimizer, train_loader, NUM_EPOCHS)


def test(model, optimizer, test_loader, num_epochs):
    model.load_state_dict(torch.load('bert.model'))
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            print(outputs)
    pass


if __name__ == '__main__':

    path = os.path.join(root_path, 'data', 'WN18RR')
    train_data = load_data(path + '/train.txt')
    valid_data = load_data(path + '/valid.txt')
    test_data = load_data(path + '/test.txt')

    tokenizer = BertTokenizer.from_pretrained(os.path.join(root_path, 'checkpoints', 'bert-base-uncased'))

    train_dataset = prepare_data(train_data,tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型和优化器
    model = BERT_KG(num_relations=11).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 定义损失函数和训练过程
    criterion = nn.CrossEntropyLoss()

    if RUN_TYPE == 'train':
        train(model, optimizer, train_loader, NUM_EPOCHS)
    else:
        test_dataset = prepare_data(test_data,tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test(model, optimizer, test_loader, num_epochs=NUM_EPOCHS)

# # 计算评估指标（这里假设是分类任务，可以根据具体任务调整评估方法）
# from sklearn.metrics import accuracy_score, classification_report
#
# accuracy = accuracy_score(true_labels, predictions)
# report = classification_report(true_labels, predictions)
#
# print(f'Accuracy: {accuracy}')
# print(f'Classification Report:\n{report}')
# # 可选：定义评估过程
# def evaluate(model, data_loader):
#     model.eval()
#     # 实现评估的逻辑，计算精度等指标
#
# # 测试模型
# # 可选：在测试集上进行评估


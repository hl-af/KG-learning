import torch
from transformers import BertTokenizer, BertModel
import argparse
"""
File: bert_base_test.py
Author: huanglei
Date: 2024-06-27

Description:
基于hugging-face的例子搭建的demo

Inputs:
- sentence 构造的一个文本
- sentences 构造的一个文本数组

Outputs:
outputs bert的结构化输出

"""
# bert基本模型测试
# 输出embedding向量
if __name__ == '__main__':
    # 加载预训练的BERT模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 假设我们有一个句子
    sentence = "This is a sample sentence."
    sentences = ['a','b','c']
    # 对句子进行编码
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # 将模型设置为评估模式
    model.eval()

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取[CLS] token的向量
    cls_embedding = outputs.pooler_output  # shape: (batch_size, hidden_size)

    print(cls_embedding.shape)  # 打印输出的形状，通常为 (1, 768)
    print(cls_embedding)  # 打印CLS向量

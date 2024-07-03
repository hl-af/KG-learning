from transformers import BertTokenizer, BertForMaskedLM
import torch
"""
File: bert_mask_test.py
Author: huanglei
Date: 2024-07-01

Description:
写mask的方式作为输入

Inputs:
- sentence 携带mask的三元组

Outputs:
predicted_token 预测结果的编码

"""
# 加载预训练的 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('checkpoints/bert-base-uncased')
model = BertForMaskedLM.from_pretrained('checkpoints/bert-base-uncased')

# 准备输入数据并掩码（mask）尾节点
subject = "Paris"
predicate = "is the capital of"
object = "[MASK]"

# 构建句子
sentence = f"{subject} {predicate} {object}"

# 使用tokenizer进行编码
inputs = tokenizer(sentence, return_tensors='pt')

# 获取[MASK]位置的索引
mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1]

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取[MASK]位置的预测分数
mask_token_logits = outputs.logits[0, mask_token_index, :]

# 获取预测结果
predicted_token_id = torch.argmax(mask_token_logits, dim=1)
predicted_token = tokenizer.decode(predicted_token_id)

print("Predicted object:", predicted_token)
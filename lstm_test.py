import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from dataset import Reader
from create_batch import get_pair_batch_train, get_pair_batch_test, toarray, get_pair_batch_train_common, toarray_float

# 假设已经有以下数据结构和参数
num_entities = 40943  # 知识图谱实体总数
num_relations = 11  # 知识图谱关系总数
embedding_dim = 100  # 嵌入维度
hidden_size = 100  # LSTM隐藏层大小
num_layers = 2  # LSTM层数


# 定义模型
class KGModel(nn.Module):
    def __init__(self):
        super(KGModel, self).__init__()
        # 实体和关系嵌入
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        # LSTM层
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)


    def forward(self, batch):
        # batch 应该包含头实体、关系和尾实体的嵌入索引
        h, r, t = batch
        h,r,t=torch.LongTensor(h),torch.LongTensor(r),torch.LongTensor(t)
        h_e = self.entity_embedding(h)
        r_e = self.relation_embedding(r)
        t_e = self.entity_embedding(t)
        # 将实体和关系嵌入拼接起来并通过LSTM
        # 这里需要具体的拼接方式，例如头实体和关系的连接等
        # 假设我们简单地将它们拼接
        input = torch.cat([h_e, r_e], dim=1)
        input = torch.cat([input,t_e],dim=1)
        x = input.view(-1, 3, embedding_dim)
        h0 = torch.zeros(num_layers * 1, x.size(0), hidden_size)  # 2 for bidirection
        c0 = torch.zeros(num_layers * 1, x.size(0), hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (B, seq_length, hidden_size*2)

        # print('out_lstm', out_lstm.shape)
        out = out.reshape(-1, hidden_size * 2 * 3)
        out = out.reshape(-1, 39 + 1, hidden_size * 2 * 3)
        out = out.reshape(-1, 39 * 2 + 2, hidden_size * 2 * 3)
        print(out[:, 0, :])
        return out[:, 0, :]


# 实例化模型
model = KGModel()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--model', default='CAGED', help='model name')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--mode', default='test', choices=['train', 'test'], help='run training or evaluation')
parser.add_argument('-ds', '--dataset', default='WN18RR', help='dataset')
args, _ = parser.parse_known_args()
parser.add_argument('--save_dir', default=f'./checkpoints/{args.dataset}/', help='model output directory')
parser.add_argument('--save_model', dest='save_model', action='store_true')
parser.add_argument('--load_model_path', default=f'./checkpoints/{args.dataset}')
parser.add_argument('--log_folder', default=f'./checkpoints/{args.dataset}/', help='model output directory')


# data
parser.add_argument('--data_path', default=f'./data/{args.dataset}/', help='path to the dataset')
parser.add_argument('--dir_emb_ent', default="entity2vec.txt", help='pretrain entity embeddings')
parser.add_argument('--dir_emb_rel', default="relation2vec.txt", help='pretrain entity embeddings')
parser.add_argument('--num_batch', default=2740, type=int, help='number of batch')
parser.add_argument('--num_train', default=0, type=int, help='number of triples')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--total_ent', default=0, type=int, help='number of entities')
parser.add_argument('--total_rel', default=0, type=int, help='number of relations')

# model architecture
parser.add_argument('--BiLSTM_input_size', default=100, type=int, help='BiLSTM input size')
parser.add_argument('--BiLSTM_hidden_size', default=100, type=int, help='BiLSTM hidden size')
parser.add_argument('--BiLSTM_num_layers', default=2, type=int, help='BiLSTM layers')
parser.add_argument('--BiLSTM_num_classes', default=1, type=int, help='BiLSTM class')
parser.add_argument('--num_neighbor', default=39, type=int, help='number of neighbors')
parser.add_argument('--embedding_dim', default=100, type=int, help='embedding dim')

# regularization
parser.add_argument('--alpha', type=float, default=0.2, help='hyperparameter alpha')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout for EaGNN')

# optimization
parser.add_argument('--max_epoch', default=6, help='max epochs')
parser.add_argument('--learning_rate', default=0.003, type=float, help='learning rate')
parser.add_argument('--gama', default=0.5, type=float, help="margin parameter")
parser.add_argument('--lam', default=0.1, type=float, help="trade-off parameter")
parser.add_argument('--mu', default=0.001, type=float, help="gated attention parameter")
parser.add_argument('--anomaly_ratio', default=0.05, type=float, help="anomaly ratio")
parser.add_argument('--num_anomaly_num', default=300, type=int, help="number of anomalies")
args = parser.parse_args()

# data_name = args.dataset
# model_name = args.model
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
dataset = Reader(args, args.data_path)
all_triples = dataset.train_data

train_idx = list(range(len(all_triples) // 2))
batch_h, batch_r, batch_t, batch_size = get_pair_batch_train_common(args, dataset, 0, train_idx,
                                                                                args.batch_size,
                                                                                args.num_neighbor)
# 假设您已经有了一个batch的数据，可以按以下方式进行前向传播

batch_data = (batch_h, batch_r, batch_t)  # 这里应该是包含头实体、关系和尾实体的嵌入索引的batch

scores = model(batch_data)

print(scores)
print('end!')
# 注意：这个代码示例假定您已经有了所需的数据预处理和批处理步骤。
# 实际使用时，您需要根据自己的数据集进行调整。
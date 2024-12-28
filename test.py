import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from models.BiLstmSAGCN import AttBiLSTM
from DataLoader.DataLoader import TraceLoader
from DataLoader.Dataset import TraceDataset
from utils.utils import label_smoothing_loss

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型参数
model = AttBiLSTM(n_classes=2,
                  feature_size=1,
                  rnn_size=6,
                  rnn_layers=2,
                  dropout=0.1,
                  num_heads=6,
                  qkv_bias=False,
                  attn_drop=0.1,
                  drop=0.1).to(device)
model.load_state_dict(torch.load('./Bestweights/BiLSTMSAGCNbest.pt'))
model.eval()  # 设置为评估模式

# 创建测试数据集和数据加载器
test_dataset = TraceDataset(path='./dataset/test_all/test_43', mode='eval', shuffle=False)
test_loader = TraceLoader(test_dataset, max_len=5000, drop_rate=0)

# 测试结果存储
all_predictions = []
all_labels = []

# 评估模型
with torch.no_grad():
    for batch_id, (data, adj) in enumerate(test_loader()):
        images = data[0].astype('float32')
        labels = data[1].astype('int64')

        image = torch.tensor(images, dtype=torch.float32).to(device)
        label = torch.tensor(labels, dtype=torch.int64).to(device)
        adj = torch.tensor(adj, dtype=torch.float32).to(device)

        predict = model(image, adj)
        all_predictions.extend(predict.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# 转换预测结果和标签为NumPy数组
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# 获取预测标签
prelabel = np.argmax(all_predictions, axis=1)
prelabel = prelabel[:, np.newaxis]

# 打印分类报告
class_result = classification_report(all_labels, prelabel, digits=4)
print(class_result)

# 如果需要计算测试损失（注意这里使用的是label smoothing loss，如果不需要可以替换为CrossEntropyLoss）
# 注意：这里的loss计算仅用于示例，实际测试时可能不需要计算loss
test_loss = label_smoothing_loss(torch.tensor(all_predictions), torch.tensor(all_labels),
                                 weight=torch.tensor([1.0, 1.0]), epsilon=0.1)
print(f'Test Loss: {test_loss.item()}')
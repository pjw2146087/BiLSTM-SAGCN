import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import time
import numpy as np
from models.BiLstmSAGCN import AttBiLSTM
from DataLoader.DataLoader import TraceLoader
from DataLoader.Dataset import TraceDataset
from utils.utils import WarmupCosineLR,label_smoothing_loss
#####################################
# Create training and evaluation datasets
train_dataset = TraceDataset(path='./dataset/sampledpaddy_2/train_43',mode='train', shuffle=True,seed_value=43)
eval_dataset = TraceDataset(path='./dataset/sampledpaddy_2/valid_43',mode='eval', shuffle=False)
train_loader = TraceLoader(train_dataset,max_len=5000,drop_rate=0)
eval_loader = TraceLoader(eval_dataset,max_len=5000,drop_rate=0)

####################################
train_losses = []
val_losses1 = []
val_losses2 = []
train_accs = []
val_accs1 = []
val_accs2 = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_test_acc = 0.0
total_train_time=0
total_eval_time=0
#torch.autograd.set_detect_anomaly(True)
maxepochs_num =90
# Set the random seed if needed
#torch.manual_seed(2023)
criterion = nn.CrossEntropyLoss().to(device)
# Initialize your model, optimizer, and LR scheduler
model = AttBiLSTM(n_classes=2,
            feature_size=1,
            rnn_size= 6,
            rnn_layers= 2,
            dropout=0.1,
            num_heads= 6,
            qkv_bias=False,
            attn_drop=0.1, 
            drop=0.1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
scheduler = WarmupCosineLR(optimizer,
                             warmup_start_lr=0.001,
                             end_lr=0.0005,
                             warmup_epochs=5,
                             total_epochs=maxepochs_num)
##########################################训练
for pass_num in range(maxepochs_num):
    model.train()
    tacc = []
    tloss = []
    correct_predictions = 0
    total_samples = 0
    print("learning-rate:", optimizer.param_groups[0]['lr'])
    
    epoch_start_time = time.time()
    for batch_id, (data, adj) in enumerate(train_loader()):
        images = data[0].astype('float32')
        labels = data[1].astype('int64')
        
        image = torch.tensor(images, dtype=torch.float32).to(device)
        label = torch.tensor(labels, dtype=torch.int64).to(device)
        adj = torch.tensor(adj, dtype=torch.float32).to(device)
        
        predict = model(image, adj)
        loss = label_smoothing_loss(predict.float(), label, weight=torch.tensor([1.0, 1.0]).to(device), epsilon=0.1)
        #loss = criterion(predict.float(),label.view(-1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        params=list(model.parameters())
        acc = (torch.argmax(predict, dim=1) == label.squeeze()).float().mean()
        correct_predictions += (torch.argmax(predict, dim=1) == label.squeeze()).sum().item()
        total_samples += label.size(0)
        tacc.append(acc.item())
        tloss.append(loss.item())
        
        if batch_id != 0 and batch_id % 50 == 0:
            print(f"train_pass: {pass_num}, batch_id: {batch_id}, train_loss: {loss.item()}, train_acc: {acc.item()}")
    
    scheduler.step()
    accuracy = correct_predictions / total_samples
    avg_train_loss = np.mean(tloss)
    train_losses.append(avg_train_loss)
    avg_train_acc = np.mean(tacc)
    train_accs.append(avg_train_acc)
    
    print(f"train: {pass_num}, batch_id: {batch_id}, train_loss: {np.mean(tloss)}, train_acc: {accuracy}")
    epoch_end_time = time.time()
    epoch_train_time = epoch_end_time - epoch_start_time
    total_train_time += epoch_train_time
    print(f"Epoch {pass_num} training time: {epoch_train_time*1000} ms")
    
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        model.eval()
        for batch_id, (data, adj) in enumerate(eval_loader()):
            
            images = data[0].astype('float32')
            labels = data[1].astype('int64')
            
            image = torch.tensor(images, dtype=torch.float32).to(device)
            label = torch.tensor(labels, dtype=torch.int64).to(device)
            adj = torch.tensor(adj, dtype=torch.float32).to(device)
            
            predict= model(image, adj)
            all_predictions.extend(predict.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        #threshold = 0.5
        #binary_predictions = (all_predictions >= threshold).astype(int)
        prelabel = np.argmax(all_predictions, axis=1)
        prelabel = prelabel[:, np.newaxis]
        class_result = classification_report(all_labels, prelabel, digits=4)
        loss1 = label_smoothing_loss(torch.tensor(all_predictions), torch.tensor(all_labels), weight=torch.tensor([1.0, 1.0]), epsilon=0.1)
        avg_acc = np.mean(np.equal(prelabel, all_labels))
        
        if avg_acc > best_test_acc:
            best_test_acc = avg_acc
            torch.save(model.state_dict(), './Bestweights/BiLSTMSAGCNbest.pt')
        
        print(class_result)
        print(f'Test: {pass_num}, Accuracy: {avg_acc:.5f}, Best: {best_test_acc:.5f}')
        
        end_time = time.time()
        epoch_eval_time = end_time - epoch_end_time
        print(f"Epoch {pass_num} evaluation time: {epoch_eval_time*1000} ms")
        total_eval_time += epoch_eval_time
        val_losses1.append(loss1)
        val_accs1.append(avg_acc)
# Calculate average training and evaluation times
avg_train_time = total_train_time / maxepochs_num
avg_eval_time = total_eval_time / maxepochs_num
print(f"Avg training time per epoch: {avg_train_time*1000} ms")
print(f"Avg evaluation time per epoch: {avg_eval_time*1000} ms")
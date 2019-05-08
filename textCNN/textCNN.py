import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
import torchtext
from tqdm import tqdm, tqdm_notebook
from nltk import word_tokenize
import random
from torch import optim
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load Corpus

text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize, fix_length=70)
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
train = torchtext.data.TabularDataset(path='train.csv', format='csv',
                                      fields={'response': ('text',text),
                                              'label': ('target',target)})
test = torchtext.data.TabularDataset(path='test.csv', format='csv',
                                     fields={'response': ('text', text)})


# Build Vocabulary

text.build_vocab(train, test, min_freq=3)


# Load Pretrained Language Model
embedding = torchtext.vocab.Vectors('wiki-news-300d-1M.vec')
tqdm_notebook().pandas() 

text.vocab.set_vectors(embedding.stoi, embedding.vectors, dim=300)

# Network
class TextCNN(nn.Module):
    
    def __init__(self, lm, padding_idx, static=True, kernel_num=128, fixed_length=50, kernel_size=[2, 5, 10], dropout=0.2):
        super(TextCNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding.from_pretrained(lm)
        if static:
            self.embedding.weight.requires_grad = False
        self.embedding.padding_idx = padding_idx
        self.conv = nn.ModuleList([nn.Conv2d(1, kernel_num, (i, self.embedding.embedding_dim)) for i in kernel_size])
        self.maxpools = [nn.MaxPool2d((fixed_length+1-i,1)) for i in kernel_size]
        self.fc = nn.Linear(len(kernel_size)*kernel_num, 15)
        
    def forward(self, input):
        x = self.embedding(input).unsqueeze(1)  # B X Ci X H X W
        x = [self.maxpools[i](torch.tanh(cov(x))).squeeze(3).squeeze(2) for i, cov in enumerate(self.conv)]  # B X Kn
        x = torch.cat(x, dim=1)  # B X Kn * len(Kz)
        y = self.fc(self.dropout(x))
        return torch.log_softmax(y,dim = 1)


# Training
def training(epoch, model, loss_func, optimizer, train_iter):
    e = 0
    
    while e < epoch:
        train_iter.init_epoch()
        losses, preds, true = [], [], []
        loss_temp = []
        for train_batch in tqdm(list(iter(train_iter)), 'epcoh {} training'.format(e)):
            model.train()
            x = train_batch.text
            y = train_batch.target.type(torch.Tensor)
            true.append(train_batch.target.numpy())
            model.zero_grad()
            pred = model.forward(x)
            loss = loss_function(pred, y.long())
            preds.append(torch.sigmoid(pred).cpu().data.numpy())
            loss.backward()
            optimizer.step()
        losses.append(np.mean(losses))
        
        e += 1
    return losses
                

# Batch Set and Train/Validation Split
random.seed(1234)
batch_size = 64
train_iter = torchtext.data.BucketIterator(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               sort=False)


# Network Init
def init_network(model, method='xavier', exclude='embedding', seed=123):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for name, w in model.named_parameters():
        if not exclude in name:
            if 'weight' in name:
                if method is 'xavier':
                    nn.init.xavier_normal_(w)
                elif method is 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.0)
            else: 
                pass

def print_model(model, ignore='embedding'):
    total = 0
    for name, w in model.named_parameters():
        if not ignore or ignore not in name:
            total += w.nelement()
            print('{} : {}  {} parameters'.format(name, w.shape, w.nelement()))
    print('-------'*4)
    print('Total {} parameters'.format(total))


text.fix_length = 70
model = TextCNN(text.vocab.vectors, padding_idx=text.vocab.stoi[text.pad_token], kernel_size=[1, 2, 3, 5], kernel_num=128, static=False, fixed_length=text.fix_length, dropout=0.1)
init_network(model,method='kaiming')
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()
print_model(model, ignore=None)

losses = training(50, model, loss_function, optimizer, train_iter)


# Predict

def predict(model, test_list):
    pred = []
    with torch.no_grad():
        for test_batch in test_list:
            model.eval()
            x = test_batch.text
            pred += list(np.argmax(model.forward(x).detach().numpy(),axis=1))
    return pred

test_list = list(torchtext.data.BucketIterator(dataset=test,
                                    batch_size=batch_size,
                                    sort=False,
                                    train=False))


preds = predict(model, test_list)
y_test = pd.read_csv("test.csv")['label'].tolist()
pred = preds
print("accuracy:",accuracy_score(y_test,pred))



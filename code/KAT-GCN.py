import numpy as np
import scipy.io as sio
import math
import time
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv 
import numpy as np
import scipy.signal as signal
import scipy
from matplotlib import cm



def read_mat(path,key):  
    data= sio.loadmat(path) 
    return(data[key])


def splitlist(list1):   
    alist=[]
    a=0
    for sublist in list1:
        try:
            for i in sublist:
                alist.append(i)
        except TypeError:
            alist.append(sublist)
    for i in alist:
        if type(i)==type([]):
            a=+1
            break
    if  a==1:
        return splitlist(alist)
    if a==0:
        return alist

def arr_size(arr,size):  
    s=[]
    for i in range(0,int(len(arr))+1,size):
        c=arr[i:i+size]
        s.append(c)
    return s

#read data
#---------------------------0 load--------------------------------
path='KAT/KATData0.mat'
key='data'
data01=read_mat(path,key)
data01=data01.tolist()
datas01=data01[0:10]+data01[100:110]+data01[200:210]
key='label'
y01=read_mat(path, key)
y01=splitlist(y01)
ys01=y01[0:10]+y01[100:110]+y01[200:210]

#---------------------------1 load--------------------------------
path='KAT/KATData1.mat'
key='data'
data02=read_mat(path,key)
data02=data02.tolist()
datas02=data02[0:10]+data02[100:110]+data02[200:210]
key='label'
y02=read_mat(path, key)
y02=splitlist(y02)
ys02=y02[0:10]+y02[100:110]+y02[200:210]

#---------------------------2 load--------------------------------
path='KAT/KATData2.mat'
key='data'
data03=read_mat(path,key)
data03=data03.tolist()
datas03=data03[0:10]+data03[100:110]+data03[200:210]
key='label'
y03=read_mat(path, key)
y03=splitlist(y03)
ys03=y03[0:10]+y03[100:110]+y03[200:210]

#---------------------------3 load--------------------------------
path='KAT/KATData3.mat'
key='data'
data04=read_mat(path,key)
data04=data04.tolist()
datas04=data04[0:10]+data04[100:110]+data04[200:210]
key='label'
y04=read_mat(path, key)
y04=splitlist(y04)
ys04=y04[0:10]+y04[100:110]+y04[200:210]

y=[] #label
for x in ys01:
    y.append(x)
for x in ys02:
    y.append(x)
for x in ys03:
    y.append(x)
for x in ys04:
    y.append(x)
y=[x-1 for x in y]   
dataa=[datas01,datas02,datas03,datas04]


def cala(data1):   #obatin spatial-temporal graph
    def calpkm(data):
        data = np.array(data)
        freqs, times, Sxx = signal.stft(data, fs=25000, window='hanning',
                                       nperseg=64, noverlap=0)
        pkm = np.zeros((len(Sxx),len(Sxx[0]),len(Sxx[0][0])))
        pkmn = np.zeros((len(Sxx),len(Sxx[0]),len(Sxx[0][0])))
        for k in range(len(Sxx)):
            for i in range(len(Sxx[0])):
                for j in range(len(Sxx[0][0])):
                    pkm[k][i][j] = abs(Sxx[k][i][j])**2/64
        return pkm

    def weight(matrix):
        W = np.zeros((len(matrix),len(matrix)))
        for i in range(len(matrix)):
            for j in range(i,len(matrix)):
                for k in range(len(matrix[0])):
                    W[i][j] += np.sqrt(np.square(matrix[i][k] - matrix[j][k]))
                W[j][i] = W[i][j]
        return W
                
    def calDifLaplacian(W1):
        n = len(W1)
        D = np.zeros((n,n))
        I = np.identity(n)
        for i in range(n):
            D[i][i] = sum(W1[i])
        L = D - W1            #Laplacian matrix
        return L
    
    p01km = calpkm(data1)
    A= []
    for k in range(len(p01km)):
        w01 = weight(p01km[k])
        L01 = calDifLaplacian(w01)
        a01,b01 = np.linalg.eig(L01) #Orthogonal decomposition
        a01 = a01.tolist()
        A.append(a01)
    return A

A = []
for x in dataa:
    A.append(cala(x))  



a = np.zeros((len(A)*len(A[0]),len(A[0][0]))) 

m = -1
for k in range(len(A)):
    for i in range(len(A[0])):
        m +=1
        for j in range(len(A[0][0])):
            a[m][j] = A[k][i][j]  



x = a
x=torch.tensor(x) #feature matrix of SuperGraph
x = x.float()


edge_index=[[],[]] #edge connection
for i in range(0,len(a),10):
    for j in range(i,i+10):
        for k in range(i,i+10):
            if j!=k:
                edge_index[0].append(j) 
                edge_index[1].append(k)


n = len(x) 
edge_index = torch.tensor(edge_index)
edge_index=edge_index.long()

y=torch.tensor(y)
y=y.long()


data = Data(edge_index=edge_index,x=x,y=y)
print(data)

data.test_mask=torch.rand(n)
data.train_mask=torch.rand(n)
data.val_mask=torch.rand(n)


arr = np.arange(n)
np.random.shuffle(arr)      #divide trian,test,validation
m = 0  
for i in arr:
    m += 1
    if m<int(n*0.4):
        data.train_mask[i] = 1
        data.test_mask[i] = 0
        data.val_mask[i] = 0
    elif int(n*0.4)<=m<int(n*0.7):
        data.train_mask[i] = 0
        data.test_mask[i] = 1
        data.val_mask[i] = 0
    else:
        data.train_mask[i]=0
        data.test_mask[i]=0
        data.val_mask[i]=1
data.test_mask=data.test_mask.bool()
data.train_mask=data.train_mask.bool()
data.val_mask=data.val_mask.bool() 





class Net(torch.nn.Module):   #GCN model
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(33, 30, K=3)
        self.conv2 = ChebConv(30, 25, K=3) 
        self.conv3 = ChebConv(25, 3, K=3)
           
          
        
    def forward(self): 
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=6e-4),
    dict(params=model.conv2.parameters(), weight_decay=3e-4),
    dict(params=model.conv3.parameters(), weight_decay=1e-4),
],lr=0.01)




def train():
    model.train()
    optimizer.zero_grad() 
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step() 




@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs  



best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))


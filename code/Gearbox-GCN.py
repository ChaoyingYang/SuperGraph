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

start = time.time()
dataset=[]
k=4
def loadDatadet(infile,k):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    return dataset

#read data
#----------------------crack 0-------------------
mj=0
temp00=[]
infile=r'Gearbox\Dataset_1\LW-00\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp00.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp00[0+j:400+j])
data00=[]
data00.append(dataset[0+mj*60:10+mj*60])   

temp01=[] 
infile=r'Gearbox\Dataset_1\LW-00\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp01.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp01[0+j:400+j])
data01=[]
data01.append(dataset[10+mj*60:20+mj*60])


infile=r'Gearbox\Dataset_1\LW-00\1500-4.txt'
temp02=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp02.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp02[0+j:400+j])
data02=[]
data02.append(dataset[20+mj*60:30+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-00\1500-6.txt'
temp03=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp03.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp03[0+j:400+j])
data03=[]
data03.append(dataset[30+mj*60:40+mj*60])

infile=r'Gearbox\Dataset_1\LW-00\1500-8.txt'
temp04=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp04.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp04[0+j:400+j])
data04=[]
data04.append(dataset[40+mj*60:50+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-00\1500-10.txt'
temp05=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp05.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp05[0+j:400+j])
data05=[]
data05.append(dataset[50+mj*60:60+mj*60])
    
# #----------------------crack 5-------------------
mj=1
temp10=[]
infile=r'Gearbox\Dataset_1\LW-01\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp10.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp10[0+j:400+j])
data10=[]
data10.append(dataset[0+mj*60:10+mj*60])   

temp11=[] 
infile=r'Gearbox\Dataset_1\LW-01\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp11.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp11[0+j:400+j])
data11=[]
data11.append(dataset[10+mj*60:20+mj*60])


infile=r'Gearbox\Dataset_1\LW-01\1500-4.txt'
temp12=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp12.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp12[0+j:400+j])
data12=[]
data12.append(dataset[20+mj*60:30+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-01\1500-6.txt'
temp13=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp13.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp13[0+j:400+j])
data13=[]
data13.append(dataset[30+mj*60:40+mj*60])

infile=r'Gearbox\Dataset_1\LW-01\1500-8.txt'
temp14=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp14.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp14[0+j:400+j])
data14=[]
data14.append(dataset[40+mj*60:50+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-01\1500-10.txt'
temp15=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp15.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp15[0+j:400+j])
data15=[]
data15.append(dataset[50+mj*60:60+mj*60])
    
# #----------------------crack 10-------------------
mj=2
temp20=[]
infile=r'Gearbox\Dataset_1\LW-02\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp20.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp20[0+j:400+j])
data20=[]
data20.append(dataset[0+mj*60:10+mj*60])   

temp21=[] 
infile=r'Gearbox\Dataset_1\LW-02\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp21.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp21[0+j:400+j])
data21=[]
data21.append(dataset[10+mj*60:20+mj*60])


infile=r'Gearbox\Dataset_1\LW-02\1500-4.txt'
temp22=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp22.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp22[0+j:400+j])
data22=[]
data22.append(dataset[20+mj*60:30+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-02\1500-6.txt'
temp23=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp23.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp23[0+j:400+j])
data23=[]
data23.append(dataset[30+mj*60:40+mj*60])

infile=r'Gearbox\Dataset_1\LW-02\1500-8.txt'
temp24=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp24.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp24[0+j:400+j])
data24=[]
data24.append(dataset[40+mj*60:50+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-02\1500-10.txt'
temp25=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp25.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp25[0+j:400+j])
data25=[]
data25.append(dataset[50+mj*60:60+mj*60])
    
# #----------------------crack 15-------------------
mj=3
temp30=[]
infile=r'Gearbox\Dataset_1\LW-03\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp30.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp30[0+j:400+j])
data30=[]
data30.append(dataset[0+mj*60:10+mj*60])   

temp31=[] 
infile=r'Gearbox\Dataset_1\LW-03\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp31.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp31[0+j:400+j])
data31=[]
data31.append(dataset[10+mj*60:20+mj*60])


infile=r'Gearbox\Dataset_1\LW-03\1500-4.txt'
temp32=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp32.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp32[0+j:400+j])
data32=[]
data32.append(dataset[20+mj*60:30+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-03\1500-6.txt'
temp33=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp33.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp33[0+j:400+j])
data33=[]
data33.append(dataset[30+mj*60:40+mj*60])

infile=r'Gearbox\Dataset_1\LW-03\1500-8.txt'
temp34=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp34.append(float(temp1[i][k]))

for j in range(0,4000,400):
    dataset.append(temp34[0+j:400+j])
data34=[]
data34.append(dataset[40+mj*60:50+mj*60])
    
infile=r'Gearbox\Dataset_1\LW-03\1500-10.txt'
temp35=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:4001]
for i in range(0,len(temp1)):
    temp35.append(float(temp1[i][k]))
for j in range(0,4000,400):
    dataset.append(temp35[0+j:400+j])
data35=[]
data35.append(dataset[50+mj*60:60+mj*60])

data1=[data00[0],data01[0],data02[0],data03[0],data04[0],data05[0],
       data10[0],data11[0],data12[0],data13[0],data14[0],data15[0],
       data20[0],data21[0],data22[0],data23[0],data24[0],data25[0],
       data30[0],data31[0],data32[0],data33[0],data34[0],data35[0]]
    
        

                     

def cala(data1):   #construct the spatial-temporal graph
    def calpkm(data):
        data = np.array(data)
        freqs, times, Sxx = signal.stft(data, fs=5000, window='hanning',
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
        L = D - W1
        return L
    
    p01km = calpkm(data1)
    A= []
    for k in range(len(p01km)):
        w01 = weight(p01km[k])
        L01 = calDifLaplacian(w01) 
        a01,b01 = np.linalg.eig(L01)    #obtain eigenvalues
        a01 = a01.tolist()
        A.append(a01)
    return A

A = []
for x in data1:
    A.append(cala(x))  


# data.x
a = np.zeros((len(A)*len(A[0]),len(A[0][0]))) 
m = -1
for k in range(len(A)):
    for i in range(len(A[0])):
        m +=1
        for j in range(len(A[0][0])):
            a[m][j] = A[k][i][j]  

x = a
x=torch.tensor(x)  # node attributes matrix
x = x.float()           
      




edge_index=[[],[]] #edge connection
for i in range(0,len(a),60):
    for j in range(i,i+60):
        for k in range(i,i+60):
            if j!=k:
                edge_index[0].append(j) 
                edge_index[1].append(k)


n = len(x) 
edge_index = torch.tensor(edge_index)
edge_index=edge_index.long()


 


y=torch.rand(len(x)) #label
for i in range(len(x)):
        if 0<=i<60:
            y[i] = 0
        if 60<=i<120:
            y[i] = 1        
        if 120<=i<180:
            y[i] = 2
        if 180<=i<240:
            y[i] = 3  
y=y.long()




data = Data(edge_index=edge_index,x=x,y=y)
print(data)



data.test_mask=torch.rand(n)
data.train_mask=torch.rand(n)
data.val_mask=torch.rand(n)


arr = np.arange(n)
np.random.shuffle(arr)   #divide train,test,validation


m = 0  
for i in arr:
    m += 1
    if m<int(n*0.3):
        data.train_mask[i] = 1
        data.test_mask[i] = 0
        data.val_mask[i] = 0
    elif int(n*0.3)<=m<int(n*0.65):
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



class Net(torch.nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(33, 25, K=3)
        self.conv2 = ChebConv(25, 16, K=3) 
        self.conv3 = ChebConv(16, 4, K=3)
           
          
        
    def forward(self): 
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=6e-4),
    dict(params=model.conv2.parameters(), weight_decay=3e-4),
    dict(params=model.conv3.parameters(), weight_decay=1e-4),
],lr=0.01)

step = [140]
base_lr = 1e-2
def adjust_lr(epoch):
    lr = base_lr * (0.1 ** np.sum(epoch >= np.array(step)))
    for params_group in optimizer.param_groups:
        params_group['lr'] = lr
    
    return lr

def train():
    model.train()
    adjust_lr(epoch)
    optimizer.zero_grad() 
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step() 
    vec = loss.cpu()
    vec1 = vec.detach().numpy()
    return vec1



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
Loss4= []
for epoch in range(1, 201):
    Loss4.append(train())
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

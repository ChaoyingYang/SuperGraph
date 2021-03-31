#Import the used modules

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



def read_mat(path,key):     #read .mat file
    data= sio.loadmat(path) 
    return(data[key])

def splitlist(list1):       #Turn a nested list into a list
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

def arr_size(arr,size):     #Divide the array into array blocks of the specified size
    s=[]
    for i in range(0,int(len(arr))+1,size):
        c=arr[i:i+size]
        s.append(c)
    return s

#Read data
#---------------------------0 load--------------------------------
path='CWRU/12k Drive End Bearing Fault Data/105.mat'
key='X105_DE_time'
data01=read_mat(path,key)
data01=data01.tolist()
data01=splitlist(data01)
data01=arr_size(data01,2048)
data01=data01[:30]

path='CWRU/12k Drive End Bearing Fault Data/118.mat'
key='X118_DE_time'
data02=read_mat(path,key)
data02=data02.tolist()
data02=splitlist(data02)
data02=arr_size(data02,2048)
data02=data02[:30]

path='CWRU/12k Drive End Bearing Fault Data/130.mat'
key='X130_DE_time'
data03=read_mat(path,key)
data03=data03.tolist()
data03=splitlist(data03)
data03=arr_size(data03,2048)
data03=data03[:30]

path='CWRU/12k Drive End Bearing Fault Data/169.mat'
key='X169_DE_time'
data04=read_mat(path,key)
data04=data04.tolist()
data04=splitlist(data04)
data04=arr_size(data04,2048)
data04=data04[:30]

path='CWRU/12k Drive End Bearing Fault Data/185.mat'
key='X185_DE_time'
data05=read_mat(path,key)
data05=data05.tolist()
data05=splitlist(data05)
data05=arr_size(data05,2048)
data05=data05[:30]

path='CWRU/12k Drive End Bearing Fault Data/197.mat'
key='X197_DE_time'
data06=read_mat(path,key)
data06=data06.tolist()
data06=splitlist(data06)
data06=arr_size(data06,2048)
data06=data06[:30]

path='CWRU/12k Drive End Bearing Fault Data/209.mat'
key='X209_DE_time'
data07=read_mat(path,key)
data07=data07.tolist()
data07=splitlist(data07)
data07=arr_size(data07,2048)
data07=data07[:30]

path='CWRU/12k Drive End Bearing Fault Data/222.mat'
key='X222_DE_time'
data08=read_mat(path,key)
data08=data08.tolist()
data08=splitlist(data08)
data08=arr_size(data08,2048)
data08=data08[:30]

path='CWRU/12k Drive End Bearing Fault Data/234.mat'
key='X234_DE_time'
data09=read_mat(path,key)
data09=data09.tolist()
data09=splitlist(data09)
data09=arr_size(data09,2048)
data09=data09[:30]

path='CWRU/12k Drive End Bearing Fault Data/3001.mat'
key='X056_DE_time'
data010=read_mat(path,key)
data010=data010.tolist()
data010=splitlist(data010)
data010=arr_size(data010,2048)
data010=data010[:30]

path='CWRU/12k Drive End Bearing Fault Data/3005.mat'
key='X048_DE_time'
data011=read_mat(path,key)
data011=data011.tolist()
data011=splitlist(data011)
data011=arr_size(data011,2048)
data011=data011[:30]


#---------------------------1 load---------------------------------
path='CWRU/12k Drive End Bearing Fault Data/106.mat'
key='X106_DE_time'
data11=read_mat(path,key)
data11=data11.tolist()
data11=splitlist(data11)
data11=arr_size(data11,2048)
data11=data11[:30]
   
path='CWRU/12k Drive End Bearing Fault Data/119.mat'
key='X119_DE_time'
data12=read_mat(path,key)
data12=data12.tolist()
data12=splitlist(data12)
data12=arr_size(data12,2048)
data12=data12[:30]

path='CWRU/12k Drive End Bearing Fault Data/131.mat'
key='X131_DE_time'
data13=read_mat(path,key)
data13=data13.tolist()
data13=splitlist(data13)
data13=arr_size(data13,2048)
data13=data13[:30]

path='CWRU/12k Drive End Bearing Fault Data/170.mat'
key='X170_DE_time'
data14=read_mat(path,key)
data14=data14.tolist()
data14=splitlist(data14)
data14=arr_size(data14,2048)
data14=data14[:30]

path='CWRU/12k Drive End Bearing Fault Data/186.mat'
key='X186_DE_time'
data15=read_mat(path,key)
data15=data15.tolist()
data15=splitlist(data15)
data15=arr_size(data15,2048)
data15=data15[:30]

path='CWRU/12k Drive End Bearing Fault Data/198.mat'
key='X198_DE_time'
data16=read_mat(path,key)
data16=data16.tolist()
data16=splitlist(data16)
data16=arr_size(data16,2048)
data16=data16[:30]

path='CWRU/12k Drive End Bearing Fault Data/210.mat'
key='X210_DE_time'
data17=read_mat(path,key)
data17=data17.tolist()
data17=splitlist(data17)
data17=arr_size(data17,2048)
data17=data17[:30]

path='CWRU/12k Drive End Bearing Fault Data/223.mat'
key='X223_DE_time'
data18=read_mat(path,key)
data18=data18.tolist()
data18=splitlist(data18)
data18=arr_size(data18,2048)
data18=data18[:30]

path='CWRU/12k Drive End Bearing Fault Data/235.mat'
key='X235_DE_time'
data19=read_mat(path,key)
data19=data19.tolist()
data19=splitlist(data19)
data19=arr_size(data19,2048)
data19=data19[:30]

path='CWRU/12k Drive End Bearing Fault Data/3002.mat'
key='X057_DE_time'
data110=read_mat(path,key)
data110=data110.tolist()
data110=splitlist(data110)
data110=arr_size(data110,2048)
data110=data110[:30]

path='CWRU/12k Drive End Bearing Fault Data/3006.mat'
key='X049_DE_time'
data111=read_mat(path,key)
data111=data111.tolist()
data111=splitlist(data111)
data111=arr_size(data111,2048)
data111=data111[:30]


#-------------------------------------2 load-----------------------------
path='CWRU/12k Drive End Bearing Fault Data/107.mat'
key='X107_DE_time'
data21=read_mat(path,key)
data21=data21.tolist()
data21=splitlist(data21)
data21=arr_size(data21,2048)
data21=data21[:30]

path='CWRU/12k Drive End Bearing Fault Data/120.mat'
key='X120_DE_time'
data22=read_mat(path,key)
data22=data22.tolist()
data22=splitlist(data22)
data22=arr_size(data22,2048)
data22=data22[:30]

path='CWRU/12k Drive End Bearing Fault Data/132.mat'
key='X132_DE_time'
data23=read_mat(path,key)
data23=data23.tolist()
data23=splitlist(data23)
data23=arr_size(data23,2048)
data23=data23[:30]

path='CWRU/12k Drive End Bearing Fault Data/171.mat'
key='X171_DE_time'
data24=read_mat(path,key)
data24=data24.tolist()
data24=splitlist(data24)
data24=arr_size(data24,2048)
data24=data24[:30]

path='CWRU/12k Drive End Bearing Fault Data/187.mat'
key='X187_DE_time'
data25=read_mat(path,key)
data25=data25.tolist()
data25=splitlist(data25)
data25=arr_size(data25,2048)
data25=data25[:30]

path='CWRU/12k Drive End Bearing Fault Data/199.mat'
key='X199_DE_time'
data26=read_mat(path,key)
data26=data26.tolist()
data26=splitlist(data26)
data26=arr_size(data26,2048)
data26=data26[:30]

path='CWRU/12k Drive End Bearing Fault Data/211.mat'
key='X211_DE_time'
data27=read_mat(path,key)
data27=data27.tolist()
data27=splitlist(data27)
data27=arr_size(data27,2048)
data27=data27[:30]

path='CWRU/12k Drive End Bearing Fault Data/224.mat'
key='X224_DE_time'
data28=read_mat(path,key)
data28=data28.tolist()
data28=splitlist(data28)
data28=arr_size(data28,2048)
data28=data28[:30]

path='CWRU/12k Drive End Bearing Fault Data/236.mat'
key='X236_DE_time'
data29=read_mat(path,key)
data29=data29.tolist()
data29=splitlist(data29)
data29=arr_size(data29,2048)
data29=data29[:30]

path='CWRU/12k Drive End Bearing Fault Data/3003.mat'
key='X058_DE_time'
data210=read_mat(path,key)
data210=data210.tolist()
data210=splitlist(data210)
data210=arr_size(data210,2048)
data210=data210[:30]

path='CWRU/12k Drive End Bearing Fault Data/3007.mat'
key='X050_DE_time'
data211=read_mat(path,key)
data211=data211.tolist()
data211=splitlist(data211)
data211=arr_size(data211,2048)
data211=data211[:30]



# ------------------------------------------3 load ---------------------------
path='CWRU/12k Drive End Bearing Fault Data/108.mat'
key='X108_DE_time'
data31=read_mat(path,key)
data31=data31.tolist()
data31=splitlist(data31)
data31=arr_size(data31,2048)
data31=data31[:30]
   
path='CWRU/12k Drive End Bearing Fault Data/121.mat'
key='X121_DE_time'
data32=read_mat(path,key)
data32=data32.tolist()
data32=splitlist(data32)
data32=arr_size(data32,2048)
data32=data32[:30]

path='CWRU/12k Drive End Bearing Fault Data/133.mat'
key='X133_DE_time'
data33=read_mat(path,key)
data33=data33.tolist()
data33=splitlist(data33)
data33=arr_size(data33,2048)
data33=data33[:30]

path='CWRU/12k Drive End Bearing Fault Data/172.mat'
key='X172_DE_time'
data34=read_mat(path,key)
data34=data34.tolist()
data34=splitlist(data34)
data34=arr_size(data34,2048)
data34=data34[:30]

path='CWRU/12k Drive End Bearing Fault Data/188.mat'
key='X188_DE_time'
data35=read_mat(path,key)
data35=data35.tolist()
data35=splitlist(data35)
data35=arr_size(data35,2048)
data35=data35[:30]

path='CWRU/12k Drive End Bearing Fault Data/200.mat'
key='X200_DE_time'
data36=read_mat(path,key)
data36=data36.tolist()
data36=splitlist(data36)
data36=arr_size(data36,2048)
data36=data36[:30]

path='CWRU/12k Drive End Bearing Fault Data/212.mat'
key='X212_DE_time'
data37=read_mat(path,key)
data37=data37.tolist()
data37=splitlist(data37)
data37=arr_size(data37,2048)
data37=data37[:30]

path='CWRU/12k Drive End Bearing Fault Data/225.mat'
key='X225_DE_time'
data38=read_mat(path,key)
data38=data38.tolist()
data38=splitlist(data38)
data38=arr_size(data38,2048)
data38=data38[:30]
 
path='CWRU/12k Drive End Bearing Fault Data/237.mat'
key='X237_DE_time'
data39=read_mat(path,key)
data39=data39.tolist()
data39=splitlist(data39)
data39=arr_size(data39,2048)
data39=data39[:30]
 
path='CWRU/12k Drive End Bearing Fault Data/3004.mat'
key='X059_DE_time'
data310=read_mat(path,key)
data310=data310.tolist()
data310=splitlist(data310)
data310=arr_size(data310,2048)
data310=data310[:30]
 
path='CWRU/12k Drive End Bearing Fault Data/3008.mat'
key='X051_DE_time'
data311=read_mat(path,key)
data311=data311.tolist()
data311=splitlist(data311)
data311=arr_size(data311,2048)
data311=data311[:30]

dataa = [data01,data02,data03,data04,data05,data06,data07,data08,data09,data010,data011,
        data11,data12,data13,data14,data15,data16,data17,data18,data19,data110,data111,
        data21,data22,data23,data24,data25,data26,data27,data28,data29,data210,data211,
        data31,data32,data33,data34,data35,data36,data37,data38,data39,data310,data311]


#Construct a spatial-temporal graph, and calculate its graph Laplacian eigenvalue as a node representation
def cala(data1):  
    def calpkm(data):
        data = np.array(data)
        freqs, times, Sxx = signal.stft(data, fs=12000, window='hanning',
                                       nperseg=64, noverlap=0)
        pkm = np.zeros((len(Sxx),len(Sxx[0]),len(Sxx[0][0])))
        pkmn = np.zeros((len(Sxx),len(Sxx[0]),len(Sxx[0][0])))
        for k in range(len(Sxx)):
            for i in range(len(Sxx[0])):
                for j in range(len(Sxx[0][0])): 
                    pkm[k][i][j] = abs(Sxx[k][i][j])**2/64  #obatain the Short-time periodogram
        return pkm

    def weight(matrix):  # Weight matrix construction
        W = np.zeros((len(matrix),len(matrix)))
        for i in range(len(matrix)):
            for j in range(i,len(matrix)):
                for k in range(len(matrix[0])):
                    W[i][j] += np.sqrt(np.square(matrix[i][k] - matrix[j][k]))
                W[j][i] = W[i][j]
        return W
                
    def calDifLaplacian(W1):     # Claculate the graph Laplacian matrix
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
        a01,b01 = np.linalg.eig(L01)
        a01 = a01.tolist() #graph Laplacian eigenvalue 
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
x=torch.tensor(x)   #Construct the input of the model 
x = x.float()           

#edge_index 
edge_index=[[],[]]
for i in range(0,len(a),10):
    for j in range(i,i+10):
        for k in range(i,i+10):
            if j!=k:
                edge_index[0].append(j)     #Construct edges in the SuperGraph 
                edge_index[1].append(k)     # There is a starting node of an edge in edge_index[0], and the ending node of an edge is stored in edge_index[1], 
                                            # the size of edge_index[0] represents the number of edges. 
n = len(x) 
edge_index = torch.tensor(edge_index)
edge_index=edge_index.long()

n1 = len(data01)
n2 = int(n1*11)
y=torch.rand(len(x))        # Construct the label of the nodes in SuperGraph
for i in range(len(x)):
    for j in range(0,4):
        if j*n2<=i<n1+j*n2:
            y[i] = 0
        if j*n2+n1<=i<n1*2+j*n2:
            y[i] = 1        
        if j*n2+n1*2<=i<n1*3+j*n2:
            y[i] = 2
        if j*n2+n1*3<=i<n1*4+j*n2:
            y[i] = 3    
        if j*n2+n1*4<=i<n1*5+j*n2:
            y[i] = 4        
        if j*n2+n1*5<=i<n1*6+j*n2:
            y[i] = 5
        if j*n2+n1*6<=i<n1*7+j*n2:
            y[i] = 6
        if j*n2+n1*7<=i<n1*8+j*n2:
            y[i] = 7        
        if j*n2+n1*8<=i<n1*9+j*n2:
            y[i] = 8
        if j*n2+n1*9<=i<n1*10+j*n2:
            y[i] = 9
        if j*n2+n1*10<=i<n1*11+j*n2:
            y[i] = 10
y=y.long()

data = Data(edge_index=edge_index,x=x,y=y)
print(data)


data.test_mask=torch.rand(n)    #Put the mask, that is, set the testing set, training set, and validation set
data.train_mask=torch.rand(n)
data.val_mask=torch.rand(n)
arr = np.arange(n)      # randomly divide the testing set, training set, and validation set
np.random.shuffle(arr) 

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


#Build the GCN model
class Net(torch.nn.Module): 
    def __init__(self):
        super(Net, self).__init__()  
        self.conv1 = ChebConv(33, 30, K=3)
        self.conv2 = ChebConv(30, 25, K=3) 
        self.conv3 = ChebConv(25, 11, K=3)
     
    def forward(self): 
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr    # the Forward path of model
        x = F.relu(self.conv1(x, edge_index, edge_weight))  
        x = F.relu(self.conv2(x, edge_index, edge_weight))    
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([                                                   #optimizer
    dict(params=model.conv1.parameters(), weight_decay=6e-4),
    dict(params=model.conv2.parameters(), weight_decay=3e-4),
    dict(params=model.conv3.parameters(), weight_decay=1e-4),
],lr=0.015)


def train():                                                                     # Backward propagation
    model.train()
    optimizer.zero_grad() 
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step() 

@torch.no_grad()
def test():         # Test
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs  

best_val_acc = test_acc = 0
for epoch in range(1, 301):     #output the results
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

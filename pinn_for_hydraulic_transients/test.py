import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sc
from matplotlib import pyplot as plt
from network import Network
from torch.optim import lr_scheduler

np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
torch.cuda.empty_cache()

if torch.cuda.is_available():
    print('cuda available')
    device = torch.device('cuda')
else:
    print('cuda not avail')
    device = torch.device('cpu')

total_epoch = 200000

def ph_tensor1(y):
    y = y.astype(np.float32)
    Y = torch.from_numpy(y)
    YY = Y.squeeze(-1)
    return YY

def ph_tensor2(z):
    z = z.astype(np.float32)
    Z = torch.from_numpy(z).T
    return Z

pipe_D = 0.05
pipe_A = 0.25 * pipe_D ** 2 * math.pi
pipe_f = 0.015
pipe_a = 1000.
a_g = 9.806
pipe_L = 500

criterion = nn.MSELoss()
data = sc.loadmat('Case_MOC_SFM/train.mat')

H_star = data['H_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 1
V_star = data['V_star']

    # Rearrange Data
X = ph_tensor1(X_star)
T = ph_tensor1(t_star)
H_o = ph_tensor2(H_star)
V_o = ph_tensor2(V_star)

Y_o = torch.stack(torch.meshgrid(X, T)).reshape(2, -1).T
HV_o = torch.cat([H_o, V_o], 1)

n_c = 30
X_c = torch.linspace(10, pipe_L, n_c)
T_c = torch.linspace(float(T.min()), float(T.max()), 401)
Y_c = torch.stack(torch.meshgrid(X_c, T_c)).reshape(2, -1).T

testdata = sc.loadmat('Case_MOC_SFM/test.mat')
H_test = testdata['H_test']  # N x T
X_test = testdata['X_test']  # N x 1
V_test = testdata['V_test']

X_t = ph_tensor1(X_test)
H_t = ph_tensor2(H_test)
V_t = ph_tensor2(V_test)

Y_t = torch.stack(torch.meshgrid(X_t, T)).reshape(2, -1).T
HV_t = torch.cat([H_t, V_t], 0)
print (HV_t)
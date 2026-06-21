import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sc
from matplotlib import pyplot as plt
from network import Network
from torch.optim import lr_scheduler
def ph_tensor1(y):
    y = y.astype(np.float32)
    Y = torch.from_numpy(y)
    YY = Y.squeeze(-1)
    return YY


def ph_tensor2(z):
    z = z.astype(np.float32)
    z = torch.from_numpy(z).T
    z = z.flatten()[:, None]
    return z

data = sc.loadmat('RU_PG/test.mat')

X_test = data ['X_test']
T_test = data ['t_test']
Q_test = data ['F_test']
alpha_test = data ['a_test']

X_t= ph_tensor1(X_test)
T_t = ph_tensor1(T_test)
Q_t = ph_tensor2(Q_test)
alpha_t = ph_tensor2(alpha_test)
print (X_t)
print (T_t)
print (Q_t)
print (alpha_t)
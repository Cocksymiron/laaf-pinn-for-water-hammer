import torch
import matplotlib.pyplot as plt
import scipy as sc
from moc_train import ph_tensor1 as pf1
from moc_train import ph_tensor2 as pf2
from Network_MLP import mMLP
from openpyxl import Workbook
import numpy as np
from network_LAAF import DNN_LAAF


layers = [2] + [24] * 4 + [2]
model = mMLP(layers)
model_loaded = torch.load('RU_PG.pth')
model.load_state_dict(model_loaded)
model.eval()
testdata = sc.io.loadmat('RU_PG/train.mat')

X_test = testdata ['X']
T_test = testdata ['t']
Q_test = testdata ['F']
alpha_test = testdata ['a']

Xt = pf1(X_test)
tt = pf1(T_test)
Qt = pf2(Q_test)
at = pf2(alpha_test)

x00 = torch.linspace(10, 500, 30)
X = torch.stack(torch.meshgrid(Xt, tt)).reshape(2, -1).T
x_test = X[:, 0]
t_test = X[:, 1]

Q_test = Qt.flatten()[:, None]  # NT x 1
a_test = at.flatten()[:, None]  # NT x 1
Qa_test = torch.cat([Q_test, a_test], 1)

with torch.no_grad():
    Qa_pred = model(X).cpu()

# Error
error_h = torch.sqrt(torch.sum((Qa_pred[:, 0] - Qa_test[:, 0]) ** 2) / torch.sum(Qa_test[:, 0] ** 2))
print('Error : %e' % error_h)

a0 = 38
a_pred = Qa_pred[:, 1]

a_pred = a_pred + (a0 - a_pred[0])  # PINN-Ye-2022
error_a = torch.sqrt(torch.sum((Qa_test[:, 1] - a_pred) ** 2) / torch.sum(Qa_test[:, 1] ** 2))
print('Error : %e' % error_a)

Qa_pred1 = torch.stack([Qa_pred[:, 0], a_pred], 1)
DATA = torch.cat([t_test.unsqueeze(1), Qa_pred1, Qa_test], 1)
DATA = DATA.numpy()
Qa_pred = Qa_pred.numpy()

list_data = DATA.tolist()
workbook = Workbook()
sheet = workbook.active
for row in list_data:
    sheet.append(row)

workbook.save("output_moc.xlsx")


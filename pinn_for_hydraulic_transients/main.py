import math
import time
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sc
from matplotlib import pyplot as plt
from network import Network
from torch.optim import lr_scheduler
D = 0.3
A = np.pi * D**2/4
r = 800
lamda = 0.01
S0 = 0.1
k1 = 0.01
np.random.seed(1234)
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
torch.cuda.empty_cache()
step_size = 2500
total_epoch = 40000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

class PINN :
    def __init__(self, x, t, Q, alpha, x_t, t_t, Q_t, alpha_t):
        layers = [2] + [20] * 8 + [2]
        self.model = Network(layers).to(device)
        XX0, TT0 = torch.meshgrid(x, t)
        self.pipe_L = 40
        self.n_collo = 10
        x_collocation = torch.linspace(10, self.pipe_L, self.n_collo)
        ub = float(t.max())
        lb = float(t.min())
        self.t_collocation = torch.linspace(lb, ub, 401)
        XX1, TT1 = torch.meshgrid(x_collocation, self.t_collocation)

        x0 = XX0.flatten()[:, None]  # NT x 1
        t0 = TT0.flatten()[:, None]  # NT x 1

        x1 = XX1.flatten()[:, None]  # NT x 1
        t1 = TT1.flatten()[:, None]  # NT x 1

        Q0 = Q.flatten()[:, None]  # NT x 1
        alpha0 = alpha.flatten()[:, None]  # NT x 1

        # training points
        x_o = torch.cat([x0, t0], 1)
        indices0 = torch.randperm(len(x0))
        self.X_o = x_o[indices0, :].to(device)

        # residual points
        x_c = torch.cat([x1, t1], 1)
        indices1 = torch.randperm(len(x1))
        self.X_c = x_c[indices1, :].requires_grad_().to(device)

        # exact solution of training points
        Qalpha0 = torch.cat([Q0, alpha0], 1)
        self.HV = Qalpha0[indices0, :].to(device)

        # Testing points
        self.X_t = torch.stack(torch.meshgrid(x_t, t_t)).reshape(2, -1).T.to(device)
        Qt = Q_t.flatten()[:, None]  # NT x 1
        alphat = alpha_t.flatten()[:, None]  # NT x 1
        self.Qalpha_t = torch.cat([Qt, alphat], 1).to(device)

        # MSE
        self.criterion = nn.MSELoss()

        # no. iteration
        self.iter = 1

        # L-BFGS
        self.lbfgs = torch.optim.LBFGS(self.model.parameters(),
                                       lr=1., max_iter=20000, max_eval=20000,
                                       history_size=50,
                                       tolerance_grad=1e-7,
                                       tolerance_change=1.0 * np.finfo(float).eps,
                                       line_search_fn="strong_wolfe")

        # Adam
        self.adam = torch.optim.Adam(self.model.parameters())

        # lr_decay
        self.schedule = lr_scheduler.ExponentialLR(self.adam, gamma=0.9, verbose=False)
        self.step_size = step_size

        self.train_loss = []  # list_loss
        self.test_loss = []
        self.lamda_list = []

    def loss_ODE_func(self, x_co):

        Qa1 = self.model(x_co)

        Q1 = Qa1[:, 0]
        alpha1 = Qa1[:, 1]

        dq_dX = torch.autograd.grad(inputs=x_co,
                                    outputs=Q1,
                                    grad_outputs=torch.ones_like(Q1),
                                    create_graph=True)[0]
        dQ_dx = dq_dX[:, 0]
        dQ_dt = dq_dX[:, 1]

        dp_dx = torch.autograd.grad(inputs=x_co,
                                    outputs=alpha1,
                                    grad_outputs=torch.ones_like(alpha1),
                                    create_graph=True)[0]
        dP_dx = dp_dx[:, 0]
        dP_dt = dp_dx[:, 1]

        f_1 = dQ_dt - A*dP_dx/r + lamda*Q1* abs (Q1) /2/ D/ A
        f_2 =  r*0.5*(Q1/S0/(k1*alpha1))**2

        return f_1, f_2

    def loss_func(self, x_ob, hv_ob):
        HV0 = self.model(x_ob)
        pred_f1, pred_f2 = self.loss_PDE_func(self.X_c)

        loss_data = self.criterion(HV0, hv_ob)
        hv00 = torch.zeros_like(pred_f1)
        loss_PDE = self.criterion(pred_f1, hv00) + self.criterion(pred_f2, hv00)

        loss = loss_data + 0.1 * loss_PDE

        if self.iter % 100 == 0:
            print(self.iter, loss.item(), loss_data.item(), loss_PDE.item())
        self.iter = self.iter + 1

        return loss, loss_data, loss_PDE

    def closure(self):  # closure for L-BFGS
        self.lbfgs.zero_grad()
        Loss, _, _ = self.loss_func(self.X_o, self.HV)
        # self.losses.append(Loss.item())
        Loss.backward()

        return Loss

    def Train(self):

        print("Using Adam:")
        for i in range(total_epoch):
            self.model.train()
            self.adam.zero_grad()
            self.adam_lamda.zero_grad()
            
            LOSS, loss0, loss1 = self.loss_func(self.X_o, self.HV)
            self.train_loss.append(LOSS.item())
            self.lamda_list.append(self.lamda.item())

            LOSS.backward()
            self.adam.step()
            self.adam_lamda.step()

            # if self.iter % step_size == 0:
            #    self.schedule.step()

            if self.iter % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    pred_HV = self.model(self.X_t)
                    val_loss = torch.sqrt(torch.sum((pred_HV - self.HV_t)**2) / torch.sum(self.HV_t ** 2))
                    self.test_loss.append(val_loss.item())

        print("Using L-BFGS")
        self.lbfgs.step(self.closure)

        print(torch.mean(self.lamda))
        # trainloss_array = np.array(self.train_loss)
        # np.save('loss_values.npy', trainloss_array)
        # testloss_array = np.array(self.test_loss)
        # np.save('L2error_values.npy', testloss_array)
        lamda_array = np.array(self.lamda_list)
        np.save('lambda_values.npy', lamda_array)

        plt.figure(1)
        plt.semilogy(self.train_loss, label='Train Loss', color='blue')
        plt.title('Model Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_train.png', format='png')
        plt.close()

        plt.figure(2)
        plt.semilogy(self.test_loss, linestyle='--', label='Test Loss', color='orange')
        plt.title('Evaluate Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_test.png', format='png')
        plt.close()


def ph_tensor1(y):
    y = y.astype(np.float32)
    Y = torch.from_numpy(y)
    YY = Y.squeeze(-1)
    return YY

def ph_tensor2(z):
    z = z.astype(np.float32)
    Z = torch.from_numpy(z).T
    return Z
if __name__ == "__main__":
    data = sc.io.loadmat('Case_MOC_SFM/train.mat')
    X_sh = data ['X']
    T_sh = data ['t']
    Q_sh = data ['Q']
    alpha_sh = ['alpha']

    X = ph_tensor1(X_sh)
    T = ph_tensor1(T_sh)
    Q_0 = ph_tensor2(Q_sh)
    alpha_0 = ph_tensor2(alpha_sh)

    data = sc.io.loadmat('Case_MOC_SFM/train.mat')

    X_test = data ['x_test']
    T_test = data ['t_test']
    Q_test = data ['Q_test']
    alpha_test = ['alpha_test']

    X_t= ph_tensor1(X_test)
    T_t = ph_tensor1(T_test)
    Q_t = ph_tensor2(Q_test)
    alpha_t = ph_tensor2(alpha_test)


    pinn = PINN(X, T, Q_0, alpha_0, X_t, T_t, Q_t, alpha_t)
    start_time = time.time()
    pinn.Train()

    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
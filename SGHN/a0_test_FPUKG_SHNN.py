"""
@author: Ru Geng
"""
import torch
import numpy as np
import pandas as pd
import os, sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from kg_equ import func

import scipy.integrate
solver = scipy.integrate.solve_ivp
import scipy.io as scio

import random
from utils import L2_loss


target = scio.loadmat('targetm_test.mat')
dftarget_test = target['target']
input = scio.loadmat('inputm_test.mat')
dfinput_test = input['input']

data_test = dfinput_test
label_test = dftarget_test



seed=32
random.seed(seed)
np.random.seed(seed)

from parameter_parser_get import get_args
from kg_equ import func
from HNNp2_Network2 import HNNp2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print(device)

def integrate_model(model_step1, t_span, y0, t_eval, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 64).to(device)

        dx,dvt = model_step1(x.to(device))
        dx = dx.data.numpy().reshape(-1)


        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval, **kwargs)


if __name__ == "__main__":
    args = get_args()

    model_step1 = HNNp2(args,device).to(device)


    N = args.n_particle

    best =24133
    M = 20000

    tend = 40
    time = 40

    label1 = str(0) + '-hnnp2.'
    best_step = best
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_hnnp2, args.name, label1, best_step)
    model_step1.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    kwargs = {'rtol': 1e-12}
    t_eval = np.linspace(0, tend, M)
    data_testnp = data_test

    data_test = torch.tensor(data_test, requires_grad=True, dtype=torch.float32)
    label_test = torch.tensor(label_test, requires_grad=True, dtype=torch.float32)
    test_loss=[]
    pred_sol = np.zeros([data_test.shape[0], M, N * 2])
    grouth_sol = np.zeros([data_test.shape[0], M, N * 2])
    for sim in range(data_test.shape[0]):

        dx,dvt = model_step1(data_test[sim].to(device))
        test_loss_hnnp=L2_loss(dvt,label_test[sim])

        test_loss.append(test_loss_hnnp.data.numpy())



        sol = solver(func, [0, time], data_testnp[sim][0, :], t_eval=t_eval, **kwargs)
        grouth_sol[sim] = sol['y'].T

        hnn_path = integrate_model(model_step1, [0, tend], data_testnp[sim][0, :], t_eval, **kwargs)
        hnn_x = hnn_path['y'].T
        pred_sol[sim] = hnn_x


    print('test_loss_hnnp:',np.mean(np.abs(test_loss)))
    print('test_loss_hnnp_std:', np.std(test_loss))




    loss_enery=np.zeros([data_test.shape[0]])
    for sim in range(data_test.shape[0]):
        N = 32
        belta = 2

        H_true = 0
        for i in range(1, N - 1):
            H_true = H_true + (grouth_sol[sim][:, N + i] ** 2) / 2 + (
                    (grouth_sol[sim][:, i + 1] - grouth_sol[sim][:, i]) ** 2) / 2 + 0.25 * (
                             (grouth_sol[sim][:, i + 1] - grouth_sol[sim][:, i]) ** 4) / 4+ grouth_sol[sim][:, i]**2/2+ grouth_sol[sim][:, i]**4/4
        H_true1 = (grouth_sol[sim][:, N] ** 2) / 2 + (
                (grouth_sol[sim][:, 1] - grouth_sol[sim][:, 0]) ** 2) / 2 + 0.25 * (
                          (grouth_sol[sim][:, 1] - grouth_sol[sim][:, 0]) ** 4) / 4+ grouth_sol[sim][:, 0]**2/2+ grouth_sol[sim][:, 0]**4/4
        H_true2 = (grouth_sol[sim][:, 2 * N - 1] ** 2) / 2 + (
                (grouth_sol[sim][:, 0] - grouth_sol[sim][:, N - 1]) ** 2) / 2 + 0.25 * (
                          (grouth_sol[sim][:, 0] - grouth_sol[sim][:, N - 1]) ** 4) / 4+ grouth_sol[sim][:, N - 1]**2/2+ grouth_sol[sim][:, N - 1]**4/4
        H_truek = H_true + H_true1 + H_true2

        H_pred = 0
        for i in range(1, N - 1):
            H_pred = H_pred + (pred_sol[sim][:, N + i] ** 2) / 2 + (
                    (pred_sol[sim][:, i + 1] - pred_sol[sim][:, i]) ** 2) / 2 + 0.25 * (
                             (pred_sol[sim][:, i + 1] - pred_sol[sim][:, i]) ** 4) / 4  + pred_sol[sim][:, i]**2/2+pred_sol[sim][:, i]**4/4
        H_pred1 = (pred_sol[sim][:, N] ** 2) / 2 + ((pred_sol[sim][:, 1] - pred_sol[sim][:, 0]) ** 2) / 2 + 0.25 * (
                    (pred_sol[sim][:, 1] - pred_sol[sim][:, 0]) ** 4) / 4  + pred_sol[sim][:, 0]**2/2+pred_sol[sim][:, 0]**4/4
        H_pred2 = (pred_sol[sim][:, 2 * N - 1] ** 2) / 2 + (
                (pred_sol[sim][:, 0] - pred_sol[sim][:, N - 1]) ** 2) / 2 + 0.25 * (
                          (pred_sol[sim][:, 0] - pred_sol[sim][:, N - 1]) ** 4) / 4  + pred_sol[sim][:, N - 1]**2/2+pred_sol[sim][:, N - 1]**4/4
        H_predk = H_pred + H_pred1 + H_pred2

        loss_enery[sim]=np.mean((H_predk-H_truek)**2)
    print('enegry_loss_gnn:', np.mean(np.abs(loss_enery)))
    print('enegry_loss_gnn_std:', np.std(loss_enery))

    t_index = 18

    ################----动态图-----##################

    fig = plt.figure(figsize=(10.0, 8.0), dpi=70)
    N = 32
    print('Generating plots...')
    index = np.arange(0, M, 100)

    pos = np.linspace(0, N - 1, num=N)

    im_particle = []

    for i in range(index.shape[0]):

        im_particle.append(plt.plot(pos[0:N], grouth_sol[t_index][index[i], 0:N], 'ko', markersize=10) + plt.plot(pos[0:N],
                                                                                                           pred_sol[
                                                                                                               t_index][
                                                                                                           index[i], 0:N],
                                                                                                           'ro',
                                                                                                           markersize=10))

    print('done')
    im_ani = animation.ArtistAnimation(fig, im_particle, interval=20, repeat_delay=3000, blit=True)

    plt.xlabel('particles')
    plt.title('FPU-KG SHNN prediction')

    im_ani.save('MixFPU_shnn.gif', writer='pillow', fps=60)
    plt.show()

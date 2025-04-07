"""
@author: Ru Geng
"""
import torch
import numpy as np
import pandas as pd
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
import scipy.sparse as sp
import networkx as nx
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from utils import L2_loss
from kg_equ import func

import scipy.integrate
solver = scipy.integrate.solve_ivp



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

from MLP_Network import MLPNN
from SGHN_Network import SGHN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print(device)

def integrate_model_sghn(model_step1,t_span,edge_index, y0,t_eval, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 64).to(device)

        dx,dvt =model_step1(x.to(device),edge_index.to(device))
        dvt=dvt.data.numpy().reshape(-1)


        return dvt

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval,**kwargs)

def integrate_model(model_step1,t_span, y0,t_eval, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 64).to(device)

        dx =model_step1(x.to(device))
        dx=dx.data.numpy().reshape(-1)


        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval,**kwargs)

def integrate_model_h(model_step1,t_span, y0,t_eval, **kwargs):
    def fun(t, np_x):
        x= torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 64).to(device)

        dx,dvt =model_step1(x.to(device))
        dvt=dvt.data.numpy().reshape(-1)


        return dvt

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval,**kwargs)

def integrate_model_sym(model_step1, M, y0,N):
    x_input = torch.tensor(y0, requires_grad=True, dtype=torch.float32)


    x_pred = np.zeros([200, N*2])
    x_pred[0]=y0
    for i in range(1,200):
        y_pd =model_step1(x_input.to(device))

        x_pred[i]=y_pd.detach().numpy()
        x_input =y_pd


    return x_pred


if __name__ == "__main__":
    args = get_args()
    model_step_mlp = MLPNN(args).to(device)

    N = args.n_particle

    best_mlp =1177

    M = 20000

    tend = 40
    time = 40
    kwargs = {'rtol': 1e-12}
    t_eval = np.linspace(0, tend, M)




    label1 = str(0) + '-mlp.'
    best_step = best_mlp
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_mlp, args.name, label1, best_step)
    model_step_mlp.load_state_dict(torch.load(path1, map_location=torch.device(device)))


    data_testnp = data_test

    data_test = torch.tensor(data_test, requires_grad=True, dtype=torch.float32)
    label_test = torch.tensor(label_test, requires_grad=True, dtype=torch.float32)


    pred_sol_mlp = np.zeros([data_test.shape[0], M, N * 2])
    grouth_sol = np.zeros([data_test.shape[0], M, N * 2])

    for sim in range(data_test.shape[0]):
        sol = solver(func, [0, time], data_testnp[sim][0, :], t_eval=t_eval, **kwargs)
        grouth_sol[sim] = sol['y'].T

        hnn_path_mlp = integrate_model(model_step_mlp, [0, tend], data_testnp[sim][0, :], t_eval, **kwargs)
        hnn_x_mlp = hnn_path_mlp['y'].T
        pred_sol_mlp[sim] = hnn_x_mlp




    from HNN_Network_200 import HNN

    model_step_hnn1 = HNN(args, device).to(device)

    best_hnn = 8843

    label1 = str(0) + '-hnn200.'
    best_step1 = best_hnn
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_hnn200, args.name, label1, best_step1)
    model_step_hnn1.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    pred_sol_hnn1 = np.zeros([data_test.shape[0], M, N * 2])
    for sim in range(data_test.shape[0]):
        hnn_path_hnn1 = integrate_model(model_step_hnn1, [0, tend], data_testnp[sim][0, :], t_eval, **kwargs)
        hnn_x_hnn1 = hnn_path_hnn1['y'].T
        pred_sol_hnn1[sim] = hnn_x_hnn1



    from HNNp2_Network2 import HNNp2

    model_step_hnnp2 = HNNp2(args, device).to(device)
    best_hnnp =24133

    label1 = str(0) + '-hnnp2.'
    best_step = best_hnnp
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_hnnp2, args.name, label1, best_step)
    model_step_hnnp2.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    pred_sol_hnnp2 = np.zeros([data_test.shape[0], M, N * 2])
    for sim in range(data_test.shape[0]):
        hnn_path_hnnp2 = integrate_model_h(model_step_hnnp2, [0, tend], data_testnp[sim][0, :], t_eval, **kwargs)
        hnn_x_hnnp2 = hnn_path_hnnp2['y'].T
        pred_sol_hnnp2[sim] = hnn_x_hnnp2



    from SYMNET_Network import LASympNet,GSympNet

    model_step_GSympNet = GSympNet(args.n_particle*2, args.symp_Glayers, args.symp_Gwidth, args.symp_activation).to(device)
    best_symnetG =1401

    label1 = str(0) + '-symnetG.'
    best_step = best_symnetG
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_symnetG, args.name, label1, best_step)
    model_step_GSympNet.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    pred_sol_G = np.zeros([data_test.shape[0], 200, N * 2])
    for sim in range(data_test.shape[0]):
        hnn_path_G = integrate_model_sym(model_step_GSympNet, 200, data_testnp[sim][0, :],N)

        pred_sol_G[sim] = hnn_path_G

    model_step_LASympNet = LASympNet(args.n_particle*2, args.symp_LAlayers, args.symp_LAsublayers, args.symp_activation).to(device)
    best_symnetLA = 18

    label1 = str(0) + '-symnetLA.'
    best_step = best_symnetLA
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_symnetLA, args.name, label1, best_step)
    model_step_LASympNet.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    pred_sol_LA = np.zeros([data_test.shape[0], 200, N * 2])
    for sim in range(data_test.shape[0]):
        hnn_path_LA = integrate_model_sym(model_step_LASympNet, 200, data_testnp[sim][0, :], N)

        pred_sol_LA[sim] = hnn_path_LA


    model_step_sghn = SGHN(args, device).to(device)
    G = nx.path_graph(args.n_particle)

    W = scio.loadmat('Wdir.mat')
    W = W['W']

    adj = sp.coo_matrix(W)
    edge_index = np.vstack((adj.row, adj.col))
    edge_index = torch.Tensor(edge_index).long().to(device)
    best_sghn = 6928
    label1 = str(0) + '-sghn.'
    best_step =best_sghn
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_sghn, args.name, label1, best_step)
    model_step_sghn.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    pred_sol_sghn = np.zeros([data_test.shape[0], M, N * 2])
    for sim in range(data_test.shape[0]):
        hnn_path_sghn = integrate_model_sghn(model_step_sghn, [0, tend], edge_index, data_testnp[sim][0, :], t_eval, **kwargs)
        hnn_x_sghn = hnn_path_sghn['y'].T
        pred_sol_sghn[sim] = hnn_x_sghn


    state_mlp = np.zeros([data_test.shape[0], M])
    state_hnn1 = np.zeros([data_test.shape[0], M])

    state_symnetLA = np.zeros([data_test.shape[0], 200])
    state_symnetG = np.zeros([data_test.shape[0],200])
    state_hnnp2 = np.zeros([data_test.shape[0], M])
    state_sghn = np.zeros([data_test.shape[0], M])
    indexs = np.arange(0, M, 100)

    for sim in range(data_test.shape[0]):
        state_mlp[sim] = np.mean(((grouth_sol[sim][:, 0:32] - pred_sol_mlp[sim][:, 0:32]) ** 2), axis=1) \
                     + np.mean(((grouth_sol[sim][:, 32:64] - pred_sol_mlp[sim][:, 32:64]) ** 2), axis=1)

        state_hnn1[sim] = np.mean(((grouth_sol[sim][:, 0:32] - pred_sol_hnn1[sim][:, 0:32]) ** 2), axis=1) \
                         + np.mean(((grouth_sol[sim][:, 32:64] - pred_sol_hnn1[sim][:, 32:64]) ** 2), axis=1)


        state_symnetG[sim] = np.mean(((grouth_sol[sim][indexs, 0:32] - pred_sol_G[sim][:, 0:32]) ** 2), axis=1) \
                           + np.mean(
            ((grouth_sol[sim][indexs, 32:64] - pred_sol_G[sim][:, 32:64]) ** 2), axis=1)

        state_symnetLA[sim] = np.mean(((grouth_sol[sim][indexs, 0:32] - pred_sol_LA[sim][:, 0:32]) ** 2), axis=1) \
                           + np.mean(
            ((grouth_sol[sim][indexs, 32:64] - pred_sol_LA[sim][:, 32:64]) ** 2), axis=1)

        state_hnnp2[sim] = np.mean(((grouth_sol[sim][:, 0:32] - pred_sol_hnnp2[sim][:, 0:32]) ** 2), axis=1) \
                          + np.mean(
            ((grouth_sol[sim][:, 32:64] - pred_sol_hnnp2[sim][:, 32:64]) ** 2), axis=1)

        state_sghn[sim] = np.mean(((grouth_sol[sim][:, 0:32] - pred_sol_sghn[sim][:, 0:32]) ** 2), axis=1) \
                         + np.mean(((grouth_sol[sim][:, 32:64] - pred_sol_sghn[sim][:, 32:64]) ** 2), axis=1)




    plt.plot(t_eval,np.mean(state_mlp,axis=0), 'dodgerblue', label='NODE', linewidth=3)



    plt.plot(t_eval,np.mean(state_hnn1,axis=0), '#8ECFC9', label='HNN', linewidth=3)

    plt.plot(t_eval[indexs], np.mean(state_symnetG, axis=0), 'yellowgreen', label='G-SympNet', linewidth=3)

    plt.plot(t_eval[indexs],np.mean(state_symnetLA,axis=0), 'violet', label='LA-SympNet', linewidth=3)

    plt.plot(t_eval, np.mean(state_hnnp2, axis=0), 'gold', label='SHNN(ours)', linewidth=3)

    plt.plot(t_eval,np.mean(state_sghn,axis=0), 'tomato', label='SGHN(ours)', linewidth=3)



    plt.xlim([0, 40])
    plt.ylabel('State MSE')
    plt.xlabel('t')
    plt.title("The state mse of FPU-KG model ")
    plt.legend(fontsize=12)
    plt.savefig("fig_statemse_FPUkg_time.png")
    plt.show()



    plt.plot(t_eval,np.mean(state_sghn,axis=0), 'tomato', label='SGHN(ours)', linewidth=3)

    plt.xlim([0, 40])
    plt.ylabel('State MSE')
    plt.xlabel('t')
    plt.title("The state mse of FPU-KG model ")
    plt.legend(fontsize=12)
    plt.savefig("fig_statemse_FPUkg_time1.png")
    plt.show()



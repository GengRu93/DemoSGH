import torch
import numpy as np
import pandas as pd
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

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
from SYMNET_Network import LASympNet,GSympNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print(device)

def integrate_model(model_step1, M, y0,N):

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

    model_step1 = GSympNet(args.n_particle*2, args.symp_Glayers, args.symp_Gwidth, args.symp_activation).to(device)



    N = args.n_particle

    best =1401
    M = 20000
    index = np.arange(0, M, 100)
    #step_size=0.002

    tend = 40
    time = 40

    label1 = str(0) + '-symnetG.'
    best_step = best
    path1 = '{}/{}{}{}.tar'.format(args.save_dir_symnetG, args.name, label1, best_step)
    model_step1.load_state_dict(torch.load(path1, map_location=torch.device(device)))

    kwargs = {'rtol': 1e-12}
    t_eval = np.linspace(0, tend, M)
    data_testnp = data_test

    data = torch.tensor(data_test, requires_grad=True, dtype=torch.float32)
    data_test = data[:,-1, :]
    label_test = data[:,1:, :]

    test_loss=[]
    pred_sol = np.zeros([data_test.shape[0], 200, N * 2])
    grouth_sol = np.zeros([data_test.shape[0], 200, N * 2])
    for sim in range(data_test.shape[0]):

        dx = model_step1(data_test[sim].to(device))
        test_loss_symnetG=L2_loss(dx,label_test[sim])

        test_loss.append(test_loss_symnetG.data.numpy())



        sol = solver(func, [0, time], data_testnp[sim][0, :], t_eval=t_eval, **kwargs)
        grouth_sol[sim] = sol['y'][:,index].T

        hnn_path = integrate_model(model_step1, 200, data_testnp[sim][0, :],N)

        pred_sol[sim] = hnn_path


    print('test_loss_symnetG:',np.mean(np.abs(test_loss)))
    print('test_loss_symnetG_std:', np.std(test_loss))




    loss_enery=np.zeros([data_test.shape[0]])
    for sim in range(data_test.shape[0]):
        N = 32
        belta = 2

        H_true = 0
        for i in range(1, N - 1):
            H_true = H_true + (grouth_sol[sim][:, N + i] ** 2) / 2 + (
                    (grouth_sol[sim][:, i + 1] - grouth_sol[sim][:, i]) ** 2) / 2 + 0.25 * (
                             (grouth_sol[sim][:, i + 1] - grouth_sol[sim][:, i]) ** 4) / 4 + grouth_sol[sim][:,
                                                                                             i] ** 2 / 2 + grouth_sol[
                                                                                                               sim][:,
                                                                                                           i] ** 4 / 4
        H_true1 = (grouth_sol[sim][:, N] ** 2) / 2 + (
                (grouth_sol[sim][:, 1] - grouth_sol[sim][:, 0]) ** 2) / 2 + 0.25 * (
                          (grouth_sol[sim][:, 1] - grouth_sol[sim][:, 0]) ** 4) / 4 + grouth_sol[sim][:, 0] ** 2 / 2 + \
                  grouth_sol[sim][:, 0] ** 4 / 4
        H_true2 = (grouth_sol[sim][:, 2 * N - 1] ** 2) / 2 + (
                (grouth_sol[sim][:, 0] - grouth_sol[sim][:, N - 1]) ** 2) / 2 + 0.25 * (
                          (grouth_sol[sim][:, 0] - grouth_sol[sim][:, N - 1]) ** 4) / 4 + grouth_sol[sim][:,
                                                                                          N - 1] ** 2 / 2 + grouth_sol[
                                                                                                                sim][:,
                                                                                                            N - 1] ** 4 / 4
        H_truek = H_true + H_true1 + H_true2

        H_pred = 0
        for i in range(1, N - 1):
            H_pred = H_pred + (pred_sol[sim][:, N + i] ** 2) / 2 + (
                    (pred_sol[sim][:, i + 1] - pred_sol[sim][:, i]) ** 2) / 2 + 0.25 * (
                             (pred_sol[sim][:, i + 1] - pred_sol[sim][:, i]) ** 4) / 4 + pred_sol[sim][:, i] ** 2 / 2 + \
                     pred_sol[sim][:, i] ** 4 / 4
        H_pred1 = (pred_sol[sim][:, N] ** 2) / 2 + ((pred_sol[sim][:, 1] - pred_sol[sim][:, 0]) ** 2) / 2 + 0.25 * (
                (pred_sol[sim][:, 1] - pred_sol[sim][:, 0]) ** 4) / 4 + pred_sol[sim][:, 0] ** 2 / 2 + pred_sol[sim][:,
                                                                                                       0] ** 4 / 4
        H_pred2 = (pred_sol[sim][:, 2 * N - 1] ** 2) / 2 + (
                (pred_sol[sim][:, 0] - pred_sol[sim][:, N - 1]) ** 2) / 2 + 0.25 * (
                          (pred_sol[sim][:, 0] - pred_sol[sim][:, N - 1]) ** 4) / 4 + pred_sol[sim][:, N - 1] ** 2 / 2 + \
                  pred_sol[sim][:, N - 1] ** 4 / 4
        H_predk = H_pred + H_pred1 + H_pred2

        loss_enery[sim]=np.mean((H_predk-H_truek)**2)
    print('enegry_loss_mlp:', np.mean(np.abs(loss_enery)))
    print('enegry_loss_mlp_std:', np.std(loss_enery))

    t_index = 18
    #################################################


    fig = plt.figure(figsize=(10.0, 8.0), dpi=70)
    N = 32
    print('Generating plots...')


    pos = np.linspace(0, N - 1, num=N)

    im_particle = []

    for i in range(200):

        im_particle.append(plt.plot(pos[0:N], grouth_sol[t_index][i,0:N], 'ko', markersize=10) + plt.plot(pos[0:N], pred_sol[t_index][i,0:N],'ro', markersize=10))

    print('done')
    im_ani = animation.ArtistAnimation(fig, im_particle, interval=20, repeat_delay=3000, blit=True)

    plt.xlabel('particles')
    plt.title('FPU-KG G-SympNet prediction')

    im_ani.save('MixFPU_G-SympNet.gif', writer='pillow', fps=60)
    plt.show()

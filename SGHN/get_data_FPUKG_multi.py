import torch
import numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp

import random
from kg_equ import func
import scipy.io
torch.backends.cudnn.determinstic = True


from parameter_parser_get import get_args
args = get_args()
seed=6
random.seed(seed)
np.random.seed(seed)
sample_train=50
samlpe_val=30
sample_test=20
sample=sample_train+samlpe_val+sample_test
N=32

y0 = np.zeros([1, N * 2])
for k in range(N):
    y0[0, k] = (random.random()) * np.sin(2 * np.pi * k / (2 * (N - 1)))

state=y0
for i in range(sample-1):
    y0 = np.zeros([1, N * 2])
    for k in range(N):
        y0[0, k] = (random.random()) * np.sin(2 * np.pi * k / (2 * (N - 1)))
    state1=y0
    state = np.concatenate((state, state1), axis=0)



time=40
M=20000
index=np.arange(0,M,100)

t_eval = np.linspace(0, time, M)

flag = False
kwargs = {'rtol': 1e-12}
for i in range(sample_train):
    print('train', i)
    sol = solver(func, [0, time],  state[i], t_eval=t_eval, **kwargs)

    tval = sol['t'][index]

    qp=sol['y'][:,index]
    print('qp.shape',qp.shape)


    xval = qp
    dxdt = func(tval, xval)
    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True






target_file = np.savetxt("target.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input.csv", x_input.T, delimiter=',')

print('train.shape')
print(x_target.T.shape)




flag = False
for i in range(sample_train,sample_train+samlpe_val):
    print('val',i)
    sol = solver(func, [0, time],  state[i], t_eval=t_eval, **kwargs)

    tval = sol['t'][index]

    qp=sol['y'][:,index]


    xval = qp
    dxdt = func(tval, xval)
    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True






target_file = np.savetxt("target_val.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input_val.csv", x_input.T, delimiter=',')
print('val.shape')
print(x_target.T.shape)



x_inputm =np.zeros([sample_test,200,N*2])
x_targetm=np.zeros([sample_test,200,N*2])

flag = False
k=0
for i in range(sample_train+samlpe_val,sample_train+samlpe_val+sample_test):
    print('test',i)
    sol = solver(func, [0, time],  state[i], t_eval=t_eval, **kwargs)

    tval = sol['t'][index]

    qp=sol['y'][:,index]

    x_inputm[k] = qp.T



    xval = qp
    dxdt = func(tval, xval)
    x_targetm[k] = dxdt.T
    if flag:
        x_input = np.concatenate([x_input, xval], 1)
        x_target = np.concatenate([x_target, dxdt], 1)
    else:
        x_input = xval
        x_target = dxdt
        flag = True
    k=k+1



scipy.io.savemat('targetm_test.mat', mdict={'target': x_targetm})
scipy.io.savemat('inputm_test.mat', mdict={'input': x_inputm})




target_file = np.savetxt("target_test.csv", x_target.T, delimiter=',')
input_file = np.savetxt("input_test.csv", x_input.T, delimiter=',')
print('test.shape')
print(x_target.T.shape)
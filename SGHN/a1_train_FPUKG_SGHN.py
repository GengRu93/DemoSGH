"""
@author: Ru Geng
"""
import time
import torch
import numpy as np
import pandas as pd
import os, sys
from SGHN_Network import SGHN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
import scipy.integrate
import scipy.sparse as sp
solve_ivp = scipy.integrate.solve_ivp
from utils import L2_loss
import utils
from parameter_parser_get import get_args
import glob
import scipy.integrate
import scipy.io as scio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




def train(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)




    W = scio.loadmat('Wdir.mat')

    W = W['W']



    adj = sp.coo_matrix(W)

    edge_index = np.vstack((adj.row, adj.col))
    edge_index = torch.Tensor(edge_index).long()


    dftarget = pd.read_csv("target.csv", header=None, dtype=np.float32)
    dfinput = pd.read_csv("input.csv", header=None, dtype=np.float32)

    dftarget_val = pd.read_csv("target_val.csv", header=None, dtype=np.float32)
    dfinput_val = pd.read_csv("input_val.csv", header=None, dtype=np.float32)

    data = dfinput.values
    print(data.shape[0] / 2)
    label = dftarget.values
    dxdt = torch.Tensor(label)

    data_val = dfinput_val.values
    label_val = dftarget_val.values
    dxdt_val = torch.Tensor(label_val)



    x = torch.tensor(data, requires_grad=True, dtype=torch.float32)
    x_val = torch.tensor(data_val)

    no_batches = int(x.shape[0] / args.batch_size)


    starttime = time.time()
###############################################################################################################
    for sim in range(1):
        stats = {'train_loss': [], 'test_loss': []}
        loss_values = []
        loss_values_val = []
        bad_counter = 0
        best_step = 0
        best = args.total_steps1 + 1
        model_step1 = SGHN(args, device).to(device)


        print("Num. of params: {:d}".format(utils.get_parameters_count(model_step1)))

        parmas = list(model_step1.parameters())
        optim1 = torch.optim.Adam(parmas, args.learn_rate2)
        optim2 = torch.optim.Adam(parmas, args.learn_rate3)

        starttime = time.time()

        for step in range(args.total_steps1 + 1):
            tt0 = time.time()
            train_loss_epoch_even = 0.0
            for batch in range(no_batches):
                ixs = torch.randperm(x.shape[0])[:args.batch_size]

                dxdt_hat, dvt_hat = model_step1(x[ixs].to(device), edge_index.to(device))

                loss = L2_loss(dxdt[ixs].to(device), dvt_hat)

                loss.backward()
                if step < 2000:
                    optim1.step()
                    optim1.zero_grad()
                else:
                    optim2.step()
                    optim2.zero_grad()
                train_loss_epoch_even += loss.item()


            stats['train_loss'].append(train_loss_epoch_even / no_batches)
            loss_values.append(train_loss_epoch_even / no_batches)

            model_step1.eval()
            dxdt_hat_val, dvt_hat_val = model_step1(x_val.to(device), edge_index.to(device))
            acc_val = L2_loss(dxdt_val.to(device), dvt_hat_val)
            loss_values_val.append(acc_val.item())

            os.makedirs(args.save_dir_sghn) if not os.path.exists(args.save_dir_sghn) else None
            label1 = str(sim)+'-sghn.'

            path1 = '{}/{}{}{}.tar'.format(args.save_dir_sghn, args.name, label1, step)

            torch.save(model_step1.state_dict(), path1)


            if loss_values_val[-1] < best:
                best = loss_values_val[-1]
                best_step = step
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience1:
                break

            files = glob.glob(args.save_dir_sghn+'/*.tar')


            for file in files:

                epoch_nb = int(file.split('.')[2])

                labell = file.split('.')[1]

                labell=labell.split('/')[2]

                if labell == 'Train'+str(sim)+'-sghn':

                    if epoch_nb < best_step:
                        os.remove(file)


            if args.verbose and step % args.print_every == 0:
                print("sim {},step {}, train_loss {:.4e},acc_loss {:.4e}| time: {:>7.12f}".format(sim, step,
                                                                                                  train_loss_epoch_even / no_batches,
                                                                                                  acc_val,
                                                                                                  (time.time() - tt0)))
        files = glob.glob(args.save_dir_sghn+'/*.tar')

        for file in files:
            epoch_nb = int(file.split('.')[2])
            labell = file.split('.')[1]
            labell = labell.split('/')[2]
            if labell == 'Train'+str(sim)+'-sghn':

                if epoch_nb > best_step:
                    os.remove(file)
        name_save_loss = './loss/SGHN_' + str(sim) + '_loss'
        path_loss = '{}.mat'.format(name_save_loss)
        scipy.io.savemat(path_loss, mdict={'loass_values': loss_values})

    print("Optimization Finished!")
    endtime = time.time()
    print("Optimization time：!")
    print(endtime - starttime)

    return model_step1


if __name__ == "__main__":
    args = get_args()
    model_step1 = train(args)


import torch
import torch.nn as nn

from models_mlp import MLP


class HNN(nn.Module):
    def __init__(self,args,device):

        super(HNN, self).__init__()
        self.args=args
        self.setup_layers()
        self.M = self.permutation_tensor(self.args.n_particle*2).to(device)


    def setup_layers(self):

        self.mlp_net =  MLP(self.args.n_particle*2, self.args.hidden_dim1, self.args.output_dim, self.args.nonlinearity)


    def forward(self, x):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            F = self.mlp_net(x)
            dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        dqp = dF @ self.M.t()

        return dqp


    def permutation_tensor(self, n):



        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])


        return -M

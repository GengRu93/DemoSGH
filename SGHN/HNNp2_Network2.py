import torch
import torch.nn as nn

from models_mlp import MLP



class HNNp2(nn.Module):
    def __init__(self,args,device):
        """Sparse version of GAT."""
        super(HNNp2, self).__init__()
        self.args=args
        self.setup_layers()
        self.M = self.permutation_tensor(self.args.n_particle*2).to(device)


    def setup_layers(self):
        """
        Setting up a sparse and a dense layer.

        """
        self.mlp_net1 =  MLP(self.args.n_particle, self.args.hidden_dim1, self.args.output_dim, self.args.nonlinearity)
        self.mlp_net2 = MLP(self.args.n_particle, self.args.hidden_dim1, self.args.output_dim, self.args.nonlinearity)

    def forward(self, x):
        x = x.requires_grad_(True)
        p = x[:, self.args.n_particle:2 * self.args.n_particle].requires_grad_(True)
        q = x[:, 0:self.args.n_particle].requires_grad_(True)
        with torch.enable_grad():
            h2 = self.mlp_net1(p)
            h1 = self.mlp_net2(q)


            F = h1 + h2
            dF = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
            dT = torch.autograd.grad(h2.sum(), p, create_graph=True)[0]
            dV = -torch.autograd.grad(h1.sum(), q, create_graph=True)[0]
        dqp = dF @ self.M.t()
        dvt = torch.cat([dT, dV], dim=1)

        return dqp, dvt

    def permutation_tensor(self, n):



        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])


        return -M

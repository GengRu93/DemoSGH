"""
@author: Ru Geng
"""
import torch
import torch.nn as nn

from models_mlp import MLP

from torch_scatter import scatter_add

class SGHN(nn.Module):
    def __init__(self,args,device):

        super(SGHN, self).__init__()
        self.args=args
        self.setup_layers()
        self.M = self.permutation_tensor(self.args.n_particle*2).to(device)



    def setup_layers(self):


        self.mlp_node_embed_update = MLP(5, 5, 5, self.args.nonlinearity)


        self.mlp_edge_embed_inti1 = MLP(1, 5, 5, self.args.nonlinearity)

        self.mlp_edge_embed_updae1 = MLP(5, 5, 5, self.args.nonlinearity)

        self.mlp_edge_embed_updae2 = MLP(5, 5, 5, self.args.nonlinearity)
        self.mlp_net2 = MLP(self.args.n_particle, self.args.hidden_dim2, self.args.output_dim, self.args.nonlinearity)


        self.V1 = MLP(5, 5, 1, self.args.nonlinearity)
        self.V2 = MLP(5, 5, 1, self.args.nonlinearity)
        self.V3 = MLP(1, 5, 1, self.args.nonlinearity)

        self.V = MLP(3, 5, 1, self.args.nonlinearity)

        self.T = MLP(1, 10, 1, self.args.nonlinearity)


    def forward(self, x, edge_index):
        x = x.requires_grad_(True)
        p=x[:, self.args.n_particle:2 *self.args.n_particle].requires_grad_(True)
        q=x[:,0:self.args.n_particle].requires_grad_(True)
        with torch.enable_grad():

            position = q.unsqueeze(2)
            velocity = p.unsqueeze(2)

            senders_idx, receivers_idx = edge_index
            node_pos_embed = position
            node_vel_embed = velocity



            eij=(q[:, senders_idx] - q[:, receivers_idx])
            eij = eij.unsqueeze(2)
            edge_embed=self.mlp_edge_embed_inti1(eij)


            node_up1 = edge_embed
            node_up2=self.mlp_node_embed_update(node_up1)
            agg_sent_edges = scatter_add(node_up2, senders_idx, dim=1, dim_size=self.args.n_particle)
            node_embed=agg_sent_edges


            node1=node_embed[:, receivers_idx, :]*node_embed[:,senders_idx,:]
            node11=self.mlp_edge_embed_updae1(node1)
            node12 = self.mlp_edge_embed_updae2(node1)
            edge_embed1=edge_embed+node11
            edge_embed2 = edge_embed + node12
            edge_embed=(edge_embed1+edge_embed2)/2

            v1=self.V1(node_embed)
            v2 = self.V2(edge_embed)
            vv=scatter_add(v2, senders_idx, dim=1, dim_size=self.args.n_particle)
            v3 = self.V3(node_pos_embed)

            Vtheta=self.V (torch.cat([v1, vv,v3], dim=2))
            h11 = Vtheta.squeeze(2)
            Vtheta = self.mlp_net2(h11)


            Ttheta=self.T(node_vel_embed)


            dT = torch.autograd.grad(Ttheta.sum(), p, create_graph=True)[0]
            dV = torch.autograd.grad(Vtheta.sum(), q, create_graph=True)[0]
        dqp = 0
        dvt=torch.cat([dT,-dV],dim=1)

        return dqp,dvt


    def permutation_tensor(self, n):



        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])


        return -M

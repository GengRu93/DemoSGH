import torch
import torch.nn as nn

from models_mlp import MLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLPNN(nn.Module):
    def __init__(self,args):

        super(MLPNN, self).__init__()
        self.args=args
        self.setup_layers()


    def setup_layers(self):

        self.mlp_net =  MLP(self.args.n_particle*2, self.args.hidden_dim_hnn, self.args.n_particle*2, self.args.nonlinearity)


    def forward(self, x):

        dqp = self.mlp_net(x)


        return dqp



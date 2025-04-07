
import torch

from utils import choose_nonlinearity
from parameter_parser_get import get_args
args = get_args()
class MLP(torch.nn.Module):

  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)





class msg(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, n_dim, gnn_hs, d, nonlinearity='tanh'):
    super(msg, self).__init__()
    self.linear1 = torch.nn.Linear(n_dim*4, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, d, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
      torch.nn.init.orthogonal_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    return self.linear4(h)



class aggr(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, n_dim, gnn_hs, d, nonlinearity='tanh'):
    super(aggr, self).__init__()
    self.linear1 = torch.nn.Linear(d+n_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, 1, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
      torch.nn.init.orthogonal_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    return self.linear4(h)

class MLP6(torch.nn.Module):

  def __init__(self, input_dim, gnn_hs, output_dim, nonlinearity='tanh'):
    super(MLP6, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, gnn_hs)
    self.linear2 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear3 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear4 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear5 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear6 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear7 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear8 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear9 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear10 = torch.nn.Linear(gnn_hs, gnn_hs)
    self.linear11 = torch.nn.Linear(gnn_hs, 1, bias=None)

    for l in [self.linear1, self.linear2, self.linear3,self.linear4, self.linear5, self.linear6,self.linear7, self.linear8, self.linear9, self.linear10, self.linear11]:
      torch.nn.init.orthogonal_(l.weight)
    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity(self.linear3(h))
    h = self.nonlinearity(self.linear4(h))
    h = self.nonlinearity(self.linear5(h))
    h = self.nonlinearity(self.linear6(h))
    h = self.nonlinearity(self.linear7(h))
    h = self.nonlinearity(self.linear8(h))
    h = self.nonlinearity( self.linear9(h) )
    h = self.nonlinearity(self.linear10(h))


    return self.linear11(h)
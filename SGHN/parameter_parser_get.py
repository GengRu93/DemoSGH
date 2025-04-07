import argparse
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--n_particle', default=32, type=int, help='The number of particles')
    parser.add_argument('--n_dim', default=1, type=int, help='Dimension per particle')
    parser.add_argument('--hidden_dim1', default=200, type=int, help='hidden dimension of mlp1')
    parser.add_argument('--hidden_dim2', default=60, type=int, help='hidden dimension of mlp2')
    parser.add_argument('--hidden_dim_hnn', default=256, type=int, help='hidden dimension of mlp_hnn')
    parser.add_argument('--output_dim', default=1, type=int, help='hidden dimension of mlp')
    parser.add_argument('--gnn_hs', default=60, type=int, help='hidden dimension of gin')
    parser.add_argument('--symp_Glayers', default=20, type=int, help='hidden dimension of gin')
    parser.add_argument('--symp_Gwidth', default=50, type=int, help='hidden dimension of gin')
    parser.add_argument('--symp_LAlayers', default=20, type=int, help='hidden dimension of gin')
    parser.add_argument('--symp_LAsublayers', default=4, type=int, help='hidden dimension of gin')
    parser.add_argument('--symp_activation', default='sigmoid', type=str, help='hidden dimension of gin')
    parser.add_argument('--d', default=40, type=int, help='output dimension of msg_net')
    parser.add_argument('--learn_rate1', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--learn_rate2', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--learn_rate3', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256,type=int, help='batch size'),
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')#silu
    parser.add_argument('--nonlinearity2', default='squareplus', type=str, help='neural net nonlinearity')  # silu
    parser.add_argument('--total_steps1', default=100000, type=int, help='number of gradient steps')
    parser.add_argument('--total_steps2', default=100000, type=int, help='number of gradient steps')
    parser.add_argument('--total_steps3', default=100000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=10, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', default='verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='Train', type=str, help='only one option right now')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--patience1', type=int, default=100, help='Patience')
    parser.add_argument('--patience2', type=int, default=100, help='Patience')
    parser.add_argument('--save_dir_hnn', default='./modelsave_hnn', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_hnn200', default='./modelsave_hnn200', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_sghn', default='./modelsave_sghn', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_hgnn', default='./modelsave_hgnn', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_hnnp', default='./modelsave_hnnp', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_hnnp2', default='./modelsave_hnnp2', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_mlp', default='./modelsave_mlp', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_symnetG', default='./modelsave_symnetG', type=str, help='where to save the trained model')
    parser.add_argument('--save_dir_symnetLA', default='./modelsave_symnetLA', type=str,
                        help='where to save the trained model')
    parser.add_argument('--save_dir_PNN', default='./modelsave_PNN', type=str,
                        help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


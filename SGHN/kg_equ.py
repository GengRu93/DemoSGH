
import numpy as np
from parameter_parser_get import get_args
args = get_args()
def func(t, y):
    dH = np.zeros_like(y)
    N=32

    alpha = 0.25
    dH[N]=y[1]-2*y[0]+y[N-1]+alpha*((y[1]-y[0])**3-(y[0]-y[N-1])**3)-y[0]- y[0]**3
    dH[0]=y[N]
    dH[2*N-1]=y[0]-2*y[N-1]+y[N-2]+alpha*((y[0]-y[N-1])**3-(y[N-1]-y[N-2])**3)-y[N-1]- y[N-1]**3
    dH[N-1]=y[2*N-1]
    for i in range(1,N-1):
        dH[N+i]=y[i+1]+y[i-1]-2*y[i]+alpha*((y[i+1]-y[i])**3-(y[i]-y[i-1])**3)-y[i]- y[i]**3
        dH[i]=y[N+i]

    return dH

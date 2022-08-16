#KFR Try-3: Like RNN+FNO, Training Data is different. For one parameter, different random points in time. 
#mapping between the (input) and (Ndt times the input)
import pandas as pd
import numpy as np
from math import erf as erf_, sin, exp as exp_
from numpy import pi, sqrt
import numba as nb
from scipy.io import savemat
import warnings
import torch
warnings.filterwarnings('ignore')


def DataGenerator_Alpha(Alpha, Samples, Ndt, InitialSolve=350, dx=207): 
    %run ./Data_Generator/KFRDG.ipynb
    X_Data = torch.zeros([Samples,dx])
    Y_Data = torch.zeros([Samples,dx])
    count = 0
    U, T_array = DataGenerator(Alpha, t=InitialSolve, CSV=False)
    for i in range(0,Samples):
    #print('i', i)
        start = int(random.uniform(-5000,-(Ndt+1)))
        X_Data[count,:] = ((torch.from_numpy(U)))[start,:].reshape(dx)
        Y_Data[count,:] = ((torch.from_numpy(U)))[start+Ndt,:].reshape(dx)
        count += 1
          
    dt = T_array[1]-T_array[0]
    
    Udict = {"input": X_Data, "u_results": Y_Data, "u_original": U, "time": T_array, "dt": dt}

        
    np.save(f'KFR_data_{Alpha}_{InitialSolve}',Udict)
    np.save(f'input_{Alpha}_{InitialSolve}',X_Data)
    np.save(f'u_results_{Alpha}_{InitialSolve}',Y_Data)
    np.save(f'u_original_{Alpha}_{InitialSolve}',U)
    
    return Udict
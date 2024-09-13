import numpy as np

def func_time_ave(listIn):
    arrayOut = np.array(listIn)
    arrayOut = np.mean(arrayOut,axis=0)
    return arrayOut

def func_calc_norm_fluc(data2d, dataAve, dataNorm, Nt, Ny, Nx):
    data2dTot = np.reshape(data2d, (Nt*Ny,Nx))
    dataAve2d = np.array([dataAve,]*(Nt*Ny))
    delt = data2dTot - dataAve2d

    sigma = np.sqrt(np.mean(delt**2,axis=0)) # rms of density fluctuations
    delt_norm = sigma/dataNorm
    return delt, delt_norm
import numpy as np
import torch
import time

def kompiouteraki(a,v_ini,u_ini,t):
    u_new = v_ini*t+(1/2)*a*t^2

def datagen(n_datas, n_samples, seed):
    #Feature
    
    #Baselines
    human_walking_acceleration_baseline = 4 # m/s^2
    base_acc_x = human_walking_acceleration_baseline/1.414
    base_acc_y = human_walking_acceleration_baseline/1.414
    acc_deviation_x = human_walking_acceleration_baseline/10
    acc_deviation_y = human_walking_acceleration_baseline/10

    print(acc_deviation_x)
    #Starting Point
    np.random.seed(seed)
    acc_start_x = np.array(np.random.normal(0, base_acc_x*2/3, size=(1, n_datas)))
    acc_start_y = np.array(np.random.normal(0, base_acc_y*2/3, size=(1, n_datas)))

    result = np.vstack([acc_start_x, acc_start_y]).T
    for j in range(n_samples-1):
        acc_x = []
        acc_y = []
        print(j)
        for i in range(n_datas):
            print(i)
            acc_x.append(np.random.normal(result[i,2*j], acc_deviation_x*2/3))
            acc_y.append(np.random.normal(result[i,2*j+1], acc_deviation_y*2/3))
        result = np.c_[result, np.array(acc_x).T, np.array(acc_y).T]

    

    #Label
    

    return result

print(datagen(10,4,1234))
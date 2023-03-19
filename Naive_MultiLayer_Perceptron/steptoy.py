import numpy as np
import math
import torch
import time

def datagen(n_datas, n_samples, noise, sigma):
    final_displacement_x = 0
    final_displacement_y = 0
    dataset = {}
    dataset['features'] = np.empty([n_datas, 1])
    for i in range(n_samples):
        step = np.array(np.random.normal(9,4/3, size=(1, n_datas))).round(decimals=0)
        angle = np.array(np.random.uniform(low=0,high=1, size=(1, n_datas)))
        
        noisy_angle = []
        temp = 0
        if noise:
            for i in range(n_datas):
                temp = np.random.normal(angle[:,i],sigma/3)
                if(temp<0):
                    temp = np.array(0)
                elif(temp > 1):
                    temp = np.array(1)
                noisy_angle.append(temp.item())

        noisy_angle = np.squeeze(np.array(noisy_angle))
        displacement_x = step*0.762*np.cos(angle*2*np.pi)
        displacement_y = step*0.762*np.sin(angle*2*np.pi)

        final_displacement_x = final_displacement_x + displacement_x
        final_displacement_y = final_displacement_y + displacement_y
        dataset['features'] = np.c_[dataset['features'], step.T, noisy_angle.T]

    #result = np.c_[step.T, angle.T, noisy_angle.T, displacement_x.T, displacement_y.T]

    dataset['features'] = dataset['features'][:,1:]
    dataset['labels'] = np.c_[final_displacement_x.T,final_displacement_y.T]

    return dataset

print(datagen(10,5,True, 0.2))

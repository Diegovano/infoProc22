import random
import time

def increment(left,right,seed):
    random.seed(seed)
    horizontal_distance = abs(left[1]-right[1])
    vertical_distance = abs(left[0]-right[0])
    horizontal_increment_max = horizontal_distance/900
    vertical_increment_max = vertical_distance/500
    horizontal_increment = random.uniform(-horizontal_increment_max, horizontal_increment_max)
    vertical_increment = random.uniform(-vertical_increment_max, vertical_increment_max)

    return horizontal_increment,vertical_increment

# for i in range(3):
#     h,v = increment((51.51157014690242, -0.19124352334865424),(51.50249343996652, -0.15188949941867913),time.time())
#     print('===')
#     print(h)
#     print(v)
#     print('===')
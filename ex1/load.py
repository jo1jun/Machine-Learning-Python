import numpy as np

def load(fileName):
    
    with open(fileName, 'r') as f:                  # file 을 read
        strings = f.readline().split(',')
        data = np.array([float(strings[0]), float(strings[1])]).reshape(1,2)    #우선 한 줄 읽고 default 값으로.
    
        for line in f.readlines():                  # open 한 file 을 한 줄씩 read
            strings = line.split(',')               # 읽은 한 줄을 , 기준으로 원소들을 분리하여 strings 에 저장
            temp = np.array([float(strings[0]), float(strings[1])]).reshape(1,2)    #원소들을 형 변환 시키고 세로로 append 하기위해 reshape
            data = np.append(data, temp, axis=0)    #세로로 append
            
    return data
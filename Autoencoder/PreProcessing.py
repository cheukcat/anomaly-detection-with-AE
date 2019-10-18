import numpy as np
import os
import sys

keys_list2 = []
keys_list3 = []

def data_convert(data, keys):
    try:
        f = float(data)
        return f
    except:
        if data in keys:
            return keys.index(data)
        else:
            keys.append(data)
            return len(keys)-1


def load_normal_data(filename):
    normal_data = []
    with open(filename, 'r') as f:#with语句自动调用close()方法
        line = f.readline()
        while line:
            eachline = line.split(',')###按行读取文本文件，每行数据以列表形式返回
            if (eachline[-2] == 'normal') and (eachline[1] == 'tcp'):
                read_data = []
                read_data.append(data_convert(eachline[2], keys_list2)) # 第2与3位的字符串报文转为label
                read_data.append(data_convert(eachline[3], keys_list3))
                for x in eachline[4:-2]:
                    read_data.append(float(x)) #读取除label以外的字符串
            #read_data = list(map(float, eachline))
                normal_data.append(read_data)
            line = f.readline()
    return normal_data

def load_attack_data(filename):
    attack_data = []
    with open(filename, 'r') as f:#with语句自动调用close()方法
        line = f.readline()
        while line:
            eachline = line.split(',')###按行读取文本文件，每行数据以列表形式返回
            if not (eachline[-2] == 'normal') and (eachline[1] == 'tcp'):
                read_data = []
                read_data.append(data_convert(eachline[2], keys_list2))
                read_data.append(data_convert(eachline[3], keys_list3))
                for x in eachline[4:-2]:
                    read_data.append(float(x)) #读取除label以外的字符串
            #read_data = list(map(float, eachline))
                attack_data.append(read_data)
            line = f.readline()
    return attack_data



if __name__ == '__main__':
    
    filename = os.path.join('../', 'NSL-KDD/' 'KDDTrain+.txt')
    filename_test = os.path.join('../', 'NSL-KDD/' 'KDDTest+.txt')

    tempStore = os.path.join('../', 'tempData')

    normal_data = load_normal_data(filename)
    np.save(os.path.join(tempStore,'tcp_normal.npy'),normal_data)

    normal_data_test = load_normal_data(filename_test)
    np.save(os.path.join(tempStore,'tcp_normal_test.npy'),normal_data_test)

    attack_data = load_attack_data(filename_test)
    np.save(os.path.join(tempStore,'tcp_attack.npy'),attack_data)

    print('The keys list on the position 2:', keys_list2)
    print('The keys list on the position 3:', keys_list3)
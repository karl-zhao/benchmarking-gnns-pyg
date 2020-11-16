import os
import re
import numpy as np
import csv
def write2csv(path):
    # path='Planetoid_node_classification/results/result_GAT_pyg_Citeseer_GPU0_23h12m32s_on_Oct_28_2020.txt'

    csv_file=open('results.csv','w',encoding='gbk',newline='')
    csv_writer=csv.writer(csv_file)
    csv_writer.writerow(['data','model','L','params','train','val','test','epoch'])
    totals = []
    for path in findAllFile(path):
        print(path)
        file=open(path)
        iterf=iter(file)
        for line in iterf:
            a = line.find('Dataset:')
            b = line.find('net_params={\'L\':')
            c=line.find('Model:')
            d=line.find('Total Parameters:')
            e=line.find('TEST ACCURACY averaged:')
            h = line.find('val ACCURACY averaged:')
            f=line.find('TRAIN ACCURACY averaged:')
            g=line.find('    Average Convergence Time (Epochs):')
            # h=line.find('params={\'seed\':')
            # print(g)
            if a == 0:
                dataset = line[line.index(':') + 2:line.index(',')]
            if b == 0:
                net = line[line.index(':') + 2:line.index(',')]
            if c == 0:
                model = line[line.index(':')+2:line.index('_')]
            if d == 0:
                Parameters = line[line.index(':')+2:line.index('\n')]
            if e == 0:
                TEST = line[line.index(':')+2:line.index('w')-1]
            if h == 0:
                val = line[line.index(':')+2:line.index('w')-1]
            if f == 0:
                TRAIN = line[line.index(':') + 2:line.index('w') - 1]
            # if h == 0:
            #     seed = line[line.index(':') + 2:line.index(',')]
            if g == 0:
                Epochs = line[line.index(':') + 2:line.index('w') - 1]
                totals.append([dataset, model, net, Parameters, TRAIN, val,TEST, Epochs])
                # csv_writer.writerow([dataset, model, net, Parameters, TRAIN, TEST, Epochs])
                break
    totals.sort(key=lambda x: ((x[0]), (x[1]), int(x[2])), reverse=False)
    out = []
    calculate = []
    for i in range(totals.__len__()):
        out.append(totals[i])
        csv_writer.writerow(out[i])
        if (i+1)%4 == 0:
            avg_train_acc = np.array(totals[i-3:i+1])[:,4]
            avg_val_acc = np.array(totals[i-3:i+1])[:,5]
            avg_test_acc = np.array(totals[i-3:i+1])[:,6]
            # avg_test_acc [totals[i-4:i][0][4], totals[:4][1][4], totals[:4][2][4], totals[:4][3][4]]
            avg_epoch = np.array(totals[i-3:i+1])[:,7]
            train_acc=str(np.around(np.mean(np.array(avg_train_acc, dtype=np.float32)),decimals=4))+'±'+str(np.around(np.std(np.array(avg_train_acc, dtype=np.float32),ddof = 1),decimals=4))
            val_acc = str(np.around(np.mean(np.array(avg_val_acc, dtype=np.float32)),decimals=4)) + '±' + str(np.around(np.std(np.array(avg_val_acc, dtype=np.float32), ddof=1),decimals=4))
            test_acc= str(np.around(np.mean(np.array(avg_test_acc, dtype=np.float32)),decimals=4))+'±'+str(np.around(np.std(np.array(avg_test_acc, dtype=np.float32),ddof = 1),decimals=4))
            Epochs_acc = str(np.around(np.mean(np.array(avg_epoch, dtype=np.float32)),decimals=4))+'±'+str(np.around(np.std(np.array(avg_epoch, dtype=np.float32),ddof = 1),decimals=4))
            calculate.append([out[i-1][0], out[i-1][1], out[i-1][2], out[i-1][3], train_acc, val_acc ,test_acc, Epochs_acc])
            csv_writer.writerow(calculate[int((i+1)/4-1)])
    csv_file.close()
    file.close()

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.txt'):
                fullname = os.path.join(root, f)
                yield fullname

def main():
    # base = 'Planetoid_node_classification/results/'SBMs_node_classification
    base = 'SBMs_node_classification/results/'
    # for path in findAllFile(base):
    #     print(path)
    np.set_printoptions(precision=4)
    write2csv(base)


if __name__ == '__main__':
    main()
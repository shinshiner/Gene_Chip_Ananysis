import numpy as np
from PCA.pca import *
import random
from copy import deepcopy
from constant import *

# 合并疾病各阶段
def merge_disease(label_dic, origin_num, annos):
    disease = ["Huntington's Disease (HD) Pathological",
               "lung adenocarcinoma (NCI_Thesaurus C0152013) has DiseaseStaging"]

    for dis in disease:
        origin_num[dis] = []
        label_dic[dis] = 0
        rubbish_key = []

        for k in label_dic:
            if dis in k and dis != k:
                label_dic[dis] += label_dic[k]
                origin_num[dis] += origin_num[k]
                rubbish_key.append(k)

        for k in rubbish_key:
            label_dic.pop(k)
            origin_num.pop(k)

        for i in range(len(annos)):
            if dis in annos[i]:
                annos[i] = dis

    return label_dic, origin_num, annos

# 提取出出现次数>=10次的疾病，返回相应基因芯片数据的序号
def look_anno():
    fin = open('data/E-TABM-185.sdrf.txt', 'r')         # 标注数据
    fout_cnt = open('output/data/anno_cnt.txt', 'w')  # 疾病的出现次数
    fout = open('output/data/data_processed.txt', 'w')  # 相应疾病对应原始数据的编号
    fout_anno = open('output/data/anno_processed.txt', 'w')

    s = fin.readline()
    annos = []
    label_dic = {}  # 各疾病出现的次数
    origin_num = {} # 存下基因芯片编号

    # 读标注
    for i in range(5896):
        s = fin.readline()
        l = s.split('\t')
        annos.append(l[7])
        if l[7] != '  ':
            if l[7] in label_dic:
                label_dic[l[7]] += 1
            else:
                label_dic[l[7]] = 1

            if l[7] in origin_num:
                origin_num[l[7]].append(i)
            else:
                origin_num[l[7]] = []
                origin_num[l[7]].append(i)

    label_dic, origin_num, annos = merge_disease(label_dic, origin_num, annos)

    # 删去出现次数少的疾病
    for k in label_dic:
        if label_dic[k] > 9:
            fout.write(k + '\t' + str(origin_num[k]) + '\n')
        else:
            origin_num.pop(k)

    # 按出现次数降序排列
    cnt_list = sorted(label_dic.items(), key=lambda item: item[1], reverse = True)
    for k in cnt_list:
        fout_cnt.write('%s: %d\n' % k)

    # 排序，此时字典变list
    label_dic = sorted(label_dic.items(), key=lambda item: item[0])
    origin_num = sorted(origin_num.items(), key=lambda item: item[0])

    # 标注顺序重组
    for k in origin_num:
        for num in k[1]:
            fout_anno.write(str(annos[num]) + '\n')

    fin.close()
    fout_cnt.close()
    fout.close()
    fout_anno.close()

    # '''for test'''
    # ftest = open('output/data/label_dic.txt', 'w')    # 各疾病出现次数
    # n = 0
    # for dd in label_dic:
    #     ftest.write("%-80s" % dd[0] + str(dd[1]) + '\n')
    #     if dd[1] > 9:
    #         n += 1
    # print('all diseases appear more than 10 times: %d' %n)
    # print('origin num:\n', origin_num)
    # ftest.close()

    return origin_num

def look_rawdata(origin_num):
    fin = open('data/microarray.original.txt', 'r')
    lines = []
    fin.readline()
    fout = open('output/pca/pca%.2f.txt' % PCA_PERCENTAGE, 'w')
    dataset_np = np.zeros([ALL_DATA, PCA[str(PCA_PERCENTAGE)]])

    # read raw data
    for i in range(22283):
        line = fin.readline()
        line = line.split('\t')
        line = line[1:]
        lines.append(list(map(float, line)))

    print("Data has been read successfully.")

    # do PCA
    data = np.array(lines).T
    print("Now reducing dimension...")
    lowDData = pca(data, PCA_PERCENTAGE)
    #print(lowDData[0][0])
    print("Finished, the new dimension is :" + str(len(lowDData[0])))

    # save pca results (.txt file and .npy)
    print("Start writing new data...")
    j = 0
    for k in origin_num:
        for num in k[1]:
            for i in range(len(lowDData[num])):
                dataset_np[j][i] = lowDData[num][i].real      # the number will be xxx+0j without .real
                fout.write(str(lowDData[num][i].real) + '\t')
            j += 1
            fout.write('\n')
    np.save('output/pca/pca%.2f.npy' % PCA_PERCENTAGE, dataset_np)
    print("Finished the whole work.")

    fin.close()
    fout.close()

    return dataset_np

# generate one-hot vectors
def anno2classes():
    fin = open('output/data/anno_processed.txt', 'r')
    fout = open('output/data/classes_label.txt', 'w')
    target_np = np.zeros([ALL_DATA, CLASSES])

    annos = fin.readlines()
    n = 0
    fout.write(str(n) + '\n') # first disease should be class 0
    target_np[0][0] = 1
    for i in range(1, ALL_DATA):
        if annos[i] != annos[i - 1]:
            n += 1
        fout.write(str(n) + '\n')
        target_np[i][n] = 1

    fin.close()
    fout.close()
    # np.save('output/data/target.npy', target_np)

    return target_np

def labeling():
    fin = open('output/data/classes_label.txt', 'r')
    target_np = np.zeros([ALL_DATA], dtype=np.int64)
    annos = fin.readlines()

    for i in range(ALL_DATA):
        target_np[i] = int(annos[i])

    return target_np

# divide training and testing data
def divide_train_test_set(dataset, targets):
    test_n = int(TEST_RADIO * ALL_DATA)
    test_idx = random.sample(range(ALL_DATA), test_n)

    train_x = np.zeros([ALL_DATA - test_n, PCA[str(PCA_PERCENTAGE)]])
    train_y = np.zeros([ALL_DATA - test_n])
    test_x = np.zeros([test_n, PCA[str(PCA_PERCENTAGE)]])
    test_y = np.zeros([test_n])

    j = 0
    k = 0
    for i in range(ALL_DATA):
        if i in test_idx:
            test_x[j] = deepcopy(dataset[i])
            test_y[j] = deepcopy(targets[i])
            j += 1
        else:
            train_x[k] = deepcopy(dataset[i])
            train_y[k] = deepcopy(targets[i])
            k += 1
    print("training set shape:", train_x.shape)
    print("testing set shape:", test_x.shape)

    np.save("output/data/dataset_train.npy", train_x)
    np.save("output/data/target_train.npy", train_y)
    np.save("output/data/dataset_test.npy", test_x)
    np.save("output/data/target_test.npy", test_y)

    print("Dataset Construction Finished !!!")

if __name__ == '__main__':
    #origin_num = look_anno()
    #dataset = look_rawdata(origin_num)
    # targets = anno2classes()
    dataset = np.load("output/pca/pca0.90.npy")
    targets = labeling()
    divide_train_test_set(dataset, targets)
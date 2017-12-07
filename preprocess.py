# 提取出出现次数>=10次的疾病，返回相应基因芯片数据的序号
def look_anno():
    fin = open('data/E-TABM-185.sdrf.txt', 'r')         # 标注数据
    fout = open('output/data/data_processed.txt', 'w')  # 相应疾病对应原始数据的编号

    s = fin.readline()
    label_dic = {}  # 各疾病出现的次数
    origin_num = {} # 存下基因芯片编号

    for i in range(5896):
        s = fin.readline()
        l = s.split('\t')
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

    for k in label_dic:
        if label_dic[k] > 9:
            fout.write(k + '\t' + str(origin_num[k]) + '\n')
        else:
            origin_num.pop(k)

    fin.close()
    fout.close()

    '''for test'''
    # ftest = open('output/data/label_dic.txt', 'w')    # 各疾病出现次数
    # label_dic = sorted(label_dic.items(), key = lambda item:item[0])
    # n = 0
    # for dd in label_dic:
    #     ftest.write("%-80s" % dd[0] + str(dd[1]) + '\n')
    #     if dd[1] > 9:
    #         n += 1
    # print('all diseases appear more than 10 times: %d' %n)
    # print('origin num:\n', origin_num)
    # ftest.close()

    return origin_num

def look_rawdata():
    fin = open('data/microarray.original.txt', 'r')




if __name__ == '__main__':
    look_anno()

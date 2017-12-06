# 提取出出现次数>=10次的疾病，返回相应基因芯片数据的序号
def look_anno():
    fin = open('data/E-TABM-185.sdrf.txt', 'r')
    fout = open('output/data/data_processed.txt', 'w')

    s = fin.readline()
    n = 0
    label_list = []
    label_dic = {}
    origin_num = []

    for i in range(5896):
        s = fin.readline()
        l = s.split('\t')
        if l[7] != '  ':
            label_list.append(l[7])

            if l[7] in label_dic:
                label_dic[l[7]] += 1
            else:
                label_dic[l[7]] = 1

    for i in range(len(label_list)):
        if label_dic[label_list[i]] > 9:
            n += 1
            fout.write('label %d' % n + '\t' + label_list[i] + '\n')
            origin_num.append(i)    # 存下基因芯片编号

    fin.close()
    fout.close()

    '''for test'''
    # n = 0
    # for dd in label_dic:
    #     if label_dic[dd] > 9:
    #         n += 1
    # print('all diseases appear more than 10 times: %d' %n)

    return origin_num

def look_rawdata():
    fin = open('data/microarray.original.txt', 'r')




if __name__ == '__main__':
    look_anno()

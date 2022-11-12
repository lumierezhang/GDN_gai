import glob


def get_feature_map(dataset):
    feature_file = open(f'./data/swat/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())#用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                                       #注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符
    return feature_list
# graph is 'fully-connect'
def get_fc_graph_struc(dataset):
    feature_file = open(f'./data/swat/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)  # (27个数据，每个数据包含26个其他节点)
    
    return struc_map

def get_prior_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    
    return struc_map


if __name__ == '__main__':
    get_graph_struc()
 
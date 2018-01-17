# coding: utf-8

# In[ ]:


import random
from math import *
import collections
from itertools import combinations
import operator

class Node(object):
    def __init__(self, data, attri_list):
        self.children = []
        self.group = []
        self.data = data
        # non leaf node
        self.attribute = None
        # self.gini = {}
        # leaf node
        self.isLeaf = False
        self.label = None
        self.attri_list = attri_list

    def attri_label(self, data):
        attribute = [x[1:] for x in data if len(x) > 1]
        labels = [x[0] for x in data]

        return attribute, labels

    def choose_attribute(self, attribute, num_attri):
        '''
        attribute: list of attribute
        num_attri: which attribtue to splited
        '''
        num = [x[int(num_attri) - int(attribute[0][0][0])][1] for x in attribute]
        return num

    def split_groups(self, nums):
        collection = list(set(nums))
        result = []

        def partition(collection):
            if len(collection) == 1:
                yield [collection]
                return

            first = collection[0]
            for smaller in partition(collection[1:]):
                for n, subset in enumerate(smaller):
                    yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
                yield [[first]] + smaller

        for i in partition(collection):
            result.append(i)
        return result

    def build_maps(self, attribute, num_attribute, label):

        groups_tmp = self.choose_attribute(attribute, num_attribute)
        groups = self.split_groups(groups_tmp)
        #collection = [x[int(num_attribute) - 1][1] for x in attribute]


        '''
            label_big_map:
            {attribute_value:{label:label_count}}
        '''
        label_big_map = {}
        label_attri = zip(label, groups_tmp)
        label_list = []
        att_list = list(set(groups_tmp))
        for i in att_list:
            label_map = {}
            for j in label_attri:
                for k in j[1]:
                    if k == i:
                        if j[0] not in label_map:
                            label_map[j[0]] = 1
                        else:
                            label_map[j[0]] += 1
            label_big_map[i] = label_map

        cnt = collections.Counter(groups_tmp)
        big_maps = {}
        for i in xrange(len(groups)):
            maps = {}
            for j in xrange(len(groups[i])):
                count = 0
                for k in groups[i][j]:
                    count += cnt[k]
                maps[j] = count
            big_maps[i] = maps

        gini = {}

        for i in xrange(len(groups)):
            big_gini_map = {}
            for j in xrange(len(groups[i])):
                gini_map = {}
                for att in groups[i][j]:
                    for la in label_big_map[att]:
                        if la not in gini_map:
                            gini_map[la] = label_big_map[att][la]
                        else:
                            gini_map[la] += label_big_map[att][la]
                big_gini_map[j] = gini_map
            gini[i] = big_gini_map
        return big_maps, gini, groups

    def calculate_gini(self, big_maps, gini):
        '''
        rtype: list of gini index of each group
        '''
        gini_val = []
        for i in big_maps:
            gini_ind = 0
            for j in gini[i]:
                gini_sum = 0
                for la in gini[i][j]:
                    p = gini[i][j][la] / float(big_maps[i][j])
                    gini_sum += p * p
                gini_ind += (1 - gini_sum) * big_maps[i][j] / float(sum(big_maps[i].values()))
            gini_val.append(gini_ind)
        return gini_val


    def split_point(self, attribute, labels, att_list):
        # print len(attribute[0])
        if len(attribute) == 0 or len(att_list) == len(attribute[0]) or len(attribute[0]) == 0:
            return (0, 0, [])
        attri_list = self.get_attribute_list(attribute, att_list)
        if len(attri_list) == 0:
            return (0, 0, [])
        point = []
        min_group = []
        for i in attri_list:
            point.append(0)
            min_group.append([])
        for i in xrange(len(attri_list)):
            big_maps, gini, groups = self.build_maps(attribute, attri_list[i], labels)
            ginis = self.calculate_gini(big_maps, gini)
            min_gini = min(ginis)
            ind = ginis.index(min_gini)
            group = groups[ind]
            min_group[i] = group
            point[i] = min_gini
        tmp = list(zip(point, attri_list, min_group))
        tmp = min(tmp, key = operator.itemgetter(0))
        gi = tmp[0]
        ind = tmp[1]
        g = tmp[2]
        return (ind, gi, g)

    def buildforest(self,data, attri_list, ratio, min_data):
            root = Node(data, attri_list)
            # nd = int(len(data) / 3)
            # new_data = random.sample(data, nd)
            new_data = self.subsample(data, ratio, min_data)
            if len(new_data) == 0:
                new_data = data
            # print ('newdata',new_data)
            attribute, label = root.attri_label(new_data)
            attri, gini, group = root.split_point(attribute, label, attri_list)
            cnt = collections.Counter(label)
            root.label = [n for n, f in cnt.most_common(1)]
            if len(group) <= 1 or len(attribute) == 0 or len(set(label)) == 1 or len(data) == 0:
                #root.data = data
                root.isLeaf = True
                return root
            root.group = group
            # attribute used to split is the attribute of the next node
            root.attribute = attri
            # saved the used attribute in each node
            used = list(root.attri_list)
            used.append(attri)
            attrs = self.choose_attribute(attribute, attri)
            children_ind = []
            for i in group:
                child = []
                for g in i:
                    for k in xrange(len(attrs)):
                            if attrs[k] == g:
                                child.append(k)
                    children_ind.append(list(set(child)))

            for i in xrange(len(label)):
                attribute[i].insert(0, label[i])
            new_data = attribute
            children = []
            for i in children_ind:
                childs = []
                for j in i:
                    childs.append(new_data[j])
                children.append(childs)
            for i in children:
                child_node = Node(i, used)
                child_node = child_node.buildforest(i, used, ratio, min_data)
                root.children.append(child_node)
            return root

    def get_attribute_list(self, attribute, att_list):
        # print attribute
        attribute_list = [x[0] for x in attribute[0]]
        for i in att_list:
            if i in attribute_list:
                attribute_list.remove(i)
        n = int(sqrt(len(attribute_list)))
        rand_attri = random.sample(attribute_list, n)
        return rand_attri

    def subsample(self, dataset, ratio, min_data):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        if n_sample < min_data:
            n_sample = min_data
        while len(sample) < n_sample:
            index = random.randrange(len(dataset))
            sample.append(dataset[index])
        return sample

class Tree():
    def __init__(self, data, att_list):
        self.node = Node(data, att_list)

    def build(self, data, attri_list, ratio, min_data):
        self.node = self.node.buildforest(data, attri_list, ratio, min_data)


    def predict(self, treeNode, new_data, trn_data):
        '''
        new_data : there is only one data inside
        '''
        # print 'data'
        target_attribute = treeNode.attribute
        group = treeNode.group
        # try:

        attribute = new_data[1:]
        #labels = [new_data[0]]
        num = attribute[int(target_attribute) - int(attribute[0][0])][1]
        child = None
        for i in xrange(len(group)):
                if num in group[i]:
                    child = treeNode.children[i]
        if child is None:
            # return self.major_voting(trn_data)
            return treeNode.label
        if child.isLeaf:
            return child.label
        else:
            return self.predict(child, new_data, trn_data)


    def predictions(self, tree, new_data, trn_data):
        result = []
        for i in new_data:
            result += self.predict(tree.node, i, trn_data)
        return result

    def get_accuracy(self, matrix, pred):
        '''
        label: unique set of label (length max)
        '''
        sums = 0
        for i in xrange(len(matrix)):
            sums += matrix[i][i]
        acc = sums / float(len(pred))
        return acc

    def confusionMatrix(self, trn_labels, pred, tst_labels):
        length = max(len(set(tst_labels)), len(set(trn_labels)))
        confusion_matrix = [[0 for i in xrange(length)] for i in xrange(length)]
        label_pair = zip(trn_labels, pred)
        for pair in label_pair:
            confusion_matrix[int(pair[0]) - 1][int(pair[1]) - 1] += 1
        return confusion_matrix

    def Random_Forest(self, train, test, num, ratio, min_data):
        '''
            train: trainning data
            test: testing data (one line data)
            num: number of trees
        '''

        def tree(train, ratio, min_data):
            tree = Tree(train, [])
            tree.build(train, [], ratio, min_data)
            return tree

        trees = [tree(train, ratio, min_data) for i in xrange(num)]
        pred = []
        for i in test:
            outputs = []
            for j in trees:
                outputs += j.predict(j.node, i, train)
            cnt = collections.Counter(outputs)
            pred += [n for n, c in cnt.most_common(1)]
        return pred


import sys
import itertools
from math import *
import time

trn = sys.argv[1]
tst = sys.argv[2]

def input_trn():
    '''
    file_name: str
    '''
    data = []
    with open(trn,'r') as f:
        for i in f:
            lines = i.strip().split(' ')
            tmp = []
            tmp.append(lines[0])
            for feature in lines[1:]:
                    ind = feature[:feature.index(':')]
                    value = feature[feature.index(':')+1:]
                    tmp.append((ind,value))
            data.append(tmp)
    return data

def input_tst():
    '''
    file_name: str
    '''
    data = []
    with open(tst,'r') as f:
        for i in f:
            lines = i.strip().split(' ')
            tmp = []
            tmp.append(lines[0])
            for feature in lines[1:]:
                    ind = feature[:feature.index(':')]
                    value = feature[feature.index(':')+1:]
                    tmp.append((ind,value))
            data.append(tmp)
    return data

def print_Matrix(confusion_matrix):
    for row in confusion_matrix:
        string = [str(int(x)) for x in row]
        print ' '.join(string)

def sorted_data(data):
    newdata = []
    for d in train:
        lab = [d[0]]
        at = sorted(d[1:], key=lambda x: x[0])
        lab += at
        newdata.append(lab)
    return data


if __name__ == '__main__':
    start_time = time.time()
    train = input_trn()
    test = input_tst()
    tree = Tree(train, [])
    tst_labels = [x[0] for x in test]
    trn_label = [x[0] for x in train]
    t = tst.split('.')
    s = '.'.join(t[:len(t) - 1])
    trainning = sorted(train)
    testing = sorted_data(test)
    if s == 'balance.scale':
        pred = tree.Random_Forest(trainning,testing, 250, 0.2, 100)
    elif s == 'nursery':
        pred = tree.Random_Forest(trainning,testing, 100,0.1,50)
    elif s == 'led':
        pred = tree.Random_Forest(trainning,testing, 150,0.4, 100)
    elif s == 'synthetic.social':
        pred = tree.Random_Forest(trainning,testing, 180,0.8, 0)
    confusion = tree.confusionMatrix(tst_labels, pred, trn_label)
    print_Matrix(confusion)
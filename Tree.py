
# coding: utf-8

# In[ ]:


from Node import *
from math import *

class Tree():
    def __init__(self, data, att_list):
        self.node = Node(data, att_list)
    
    def build(self, data, attri_list, Ttype):
        self.node = self.node.buildtree(data,attri_list, Ttype)
        
    def major_voting(self, data):
        label = [x[0] for x in data]
        cnt = collections.Counter(label)
        return [n for n,c in cnt.most_common(1)]
    
    def predict(self, treeNode, new_data, trn_data):
            '''
            new_data : there is only one data inside
            '''
            #print 'data'
            target_attribute = treeNode.attribute
            group = treeNode.group
        #try:
            
            attribute = new_data[1:]
            labels = [new_data[0]]
            num = attribute[int(target_attribute)-1][1:]
            #num = [i[1:] for i in sorted(attribute) if i[0] == target_attribute]
            child = None
            for i in xrange(len(group)):
                    #print group
                    for j in num:
                        #
                        if list(j)[0] in group[i]:
                            child = treeNode.children[i]
            # major voting
            #label_tmp = [i[0] for i in treeNode.data]
#             if child is None:
#                 cnt = collections.Counter(label_tmp)
#                 return [n for n, c in cnt.most_common(1)]
            if child is None:
                return self.major_voting(trn_data)
            if child.isLeaf:
                return child.label
            else:
                return self.predict(child, new_data, trn_data)
    
    def predictions(self, tree, new_data, trn_data):
        result = []
        for i in new_data:
            result += self.predict(tree.node,i, trn_data)
        return result
        
    def get_accuracy(self, matrix, pred):
            '''
            label: unique set of label (length max)
            '''
            sums = 0
            for i in xrange(len(matrix)):
                sums += matrix[i][i]
            acc = sums/float(len(pred))
            return acc
    
    
    def confusionMatrix(self, trn_labels, pred):
        length = max(len(set(pred)), len(set(trn_labels)))
        confusion_matrix = [[0 for i in xrange(length)] for i in xrange(length)]
        label_pair = zip(trn_labels, pred)
        for pair in label_pair:
            confusion_matrix[int(pair[0])-1][int(pair[1])-1] += 1
        return confusion_matrix
    
    def Random_Forest(self, train, test, num):
        '''
            train: trainning data
            test: testing data (one line data)
            num: number of trees
        '''
        def tree(train):
            tree = Tree(train, [])
            tree.build(train, [], 'rf')
            return tree
        
        trees = [tree(train) for i in xrange(num)]
        pred = []
        for i in test:
            outputs = []
            for j in trees:
                outputs += self.predict(j.node, i, train)
            cnt = collections.Counter(outputs)
            pred += [n for n,c in cnt.most_common(1)]
        return pred


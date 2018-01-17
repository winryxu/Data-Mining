
# coding: utf-8

# ## Question 3

# ## Answer
# ### Length of minimum occurence window

# S1: 4
# 
# S2: 4
# 
# S3: 4

# ### Number of outliers

# S1: 1
# 
# S2: 0
# 
# S3: 1

# ## Question 4

# In[129]:


from collections import Counter
import itertools
import sys


# In[219]:


lst = []
with open('Data.txt') as f:
    lst.append(f.read())


# In[220]:


lst = lst[0].split('\r\n')


# In[221]:


tmp1 = lst[0].split(', ')
theta = float(tmp1[0])
ep = float(tmp1[1])


# In[222]:


s1 = lst[1]
s2 = lst[2]
s3 = lst[3]


# In[223]:


s1 = s1.split(', ')
s2 = s2.split(', ')
s3 = s3.split(', ')


# In[224]:


tmp = [i for i in sorted(set(s1))]
pattern = []


# In[225]:


for i in xrange(1,len(tmp)+1):
    for j in itertools.combinations(tmp,i):
        pattern.append(j)


# In[226]:


c1 = Counter(s1)
c2 = Counter(s2)
c3 = Counter(s3)


# In[227]:


dic = {}
def find_seq(dic,sequent,counter):
    '''
    dic: dictionary
    sequent: str
    counter: collections.Counter
    '''
    for i in pattern:
        for j in xrange(len(i)):
            if counter[i[j]] == 0:
                break
            if j == len(i) - 1:
                if i in dic:
                    dic[i].append(sequent)
                else:
                    dic[i] = [sequent]
    return
find_seq(dic,'s1',c1)
find_seq(dic,'s2',c2)
find_seq(dic,'s3',c3)


# In[228]:


patterns = []
for i in pattern:
    if i in dic and (len(dic[i])/3.0) < theta:
        del dic[i]
    else:
        patterns.append(i)


# In[229]:


def find_sequent(sequent, pattern):
    dic1 = {}
    p_tmp = list(pattern)
    for i in xrange(len(sequent)):
        if sequent[i] in p_tmp:
            if sequent[i] in dic1:
                dic1[sequent[i]].append(i)
            else:
                dic1[sequent[i]] = [i]
    lst = []
    result = []
    end = -1
    lst = []
    for i in p_tmp:
        lst.append(dic1[i])
    prod = list(itertools.product(*lst))
    result = []
    tmp = []
    min_length = sys.maxint
    for i in prod:
        start = min(i)
        end = max(i)
        result.append(sequent[start:end+1])
        min_length = end - start
    return result   


# In[230]:


def find_outlier_length(sequent,pattern):
    min_count = 999
    for i in sequent:
        count = 0
        for j in i:
            if j not in pattern:
                count += 1
        min_count = min(count,min_count)
    return min(count,min_count)


# In[231]:


for i in patterns:
    for j in dic[i]:
        if j == 's1':
            s = find_sequent(s1, i)
            length = find_outlier_length(s, i)
        if j == 's2':
            s = find_sequent(s2, i)
            length = find_outlier_length(s, i)
        if j == 's3':
            s = find_sequent(s3, i)
            length = find_outlier_length(s, i)
        if length/float(len(i)) > ep and i in dic:
            dic[i].remove(j)


# In[232]:


for i in patterns:
    if i in dic and (len(dic[i])/3.0) < theta:
        del dic[i]


# In[233]:


output = [list(i) for i in dic.keys()]
output = sorted(output)


# In[236]:


with open("wxu41-HW3.txt", "w") as text_file:
    for i in output:
        for j in xrange(len(i)):
            if j < len(i) - 1:
                text_file.write("%s, " % i[j])
            else:
                text_file.write("%s\n" % i[j])


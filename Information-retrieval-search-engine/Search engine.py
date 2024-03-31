#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import natsort as nt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")


# # PART 1

# ### a) Read 10 files

# In[2]:


path = 'C:/Users/Eyad/Desktop/AI course,Data Science/projects/IR CS/documents/'
os.chdir(path)


# In[3]:


files = nt.natsorted(os.listdir())
files


# In[4]:


dicts = {}
keys = range(1,len(files)+1)
values = []
for file in files:
    with open(file , 'r', encoding='utf-8') as f:
        values.append(f.read())
for i in keys:
        dicts[i] = values[i-1]
print(dicts)


# ### b ) Tokenization

# In[5]:


def tokenize_query(query):
    query = word_tokenize(query)
    return query


# In[6]:


for i in dicts:
    dicts[i] = word_tokenize(dicts[i])
    print(dicts[i])


# In[7]:


#Test
q = 'Hello im eyad'
q = tokenize_query(q)
q


# ## c) Apply stop words (except: in , to,where)

# In[8]:


stopwords_edited = stopwords.words('english')
stopwords_edited.remove('in')
stopwords_edited.remove('to')
stopwords_edited.remove('where')


# In[9]:


def remove_stop_words(query):
    query = tokenize_query(query)
    query = [word for word in query if not word in stopwords_edited]
    return ' '.join(query)


# In[10]:


for i in dicts:
    dicts[i] = [w for w in dicts[i] if not w.lower() in stopwords_edited]
    print(dicts[i])


# In[11]:


dicts


# In[12]:


all_words = []
for doc in dicts:
    for term in dicts[doc]:
        all_words.append(term)
all_words = sorted(all_words)


# In[13]:


#test
q = 'Where are you at the to in from from in egypt'
q = remove_stop_words(q)
q


# # PART 2

# ### a) Building positional index

# In[14]:


doc_no = 1
pos_idx = {}


# In[15]:


for doc in dicts:
    for pos,term in enumerate(dicts[doc]):
        if term in pos_idx:
            pos_idx[term][0] += 1
            if doc_no in pos_idx[term][1]:
                pos_idx[term][1][doc_no].append(pos)
            else:
                pos_idx[term][1][doc_no] = [pos]
        else:
    
            pos_idx[term]=[]

            pos_idx[term].append(1)

            pos_idx[term].append({})

            pos_idx[term][1][doc_no] = [pos]
        
    doc_no +=1

print(pos_idx)


# In[16]:


sum_freq = sum([pos_idx[term][0] for term in set(all_words)])
sum_freq


# ### b) Allow users to write queries

# In[17]:


def return_matched_docs_ix(q):
    pos_idx_list = [[] for i in range(len(pos_idx))]
    for w in q.split():
        try:
            for k in pos_idx[w][1].keys():

                if pos_idx_list[k-1] != []:

                    if pos_idx_list[k-1][-1] == pos_idx[w][1][k][0]-1:
                        pos_idx_list[k-1].append(pos_idx[w][1][k][0])

                else:
                        pos_idx_list[k-1].append(pos_idx[w][1][k][0])

            
            for ix , lists in enumerate(pos_idx_list):
                if len(q.split()) == len(lists):
                    print("Matched in doc number:" , ix+1)
        except KeyError:
            print("No matched document -> invalid input")


# In[18]:


# test
q = "brutus and caeser"
q = remove_stop_words(q)
return_matched_docs_ix(q)


# # Part 3

# ### a) Term frequency

# In[19]:


def get_term_freq(doc):
    words_found = dict.fromkeys(all_words , 0)
    for word in doc:
        words_found[word] += 1
    return words_found   


# In[20]:


get_term_freq(dicts[1]).values()


# In[21]:


tf = pd.DataFrame()
for i in range(1 , len(dicts)+1):
    tf[i] =  pd.DataFrame(
                            get_term_freq(dicts[i]).values() ,
                            index = get_term_freq(dicts[i]).keys()
                         )
    
tf.columns = [f"DOC_{i}" for i in range(1,11)]
tf.style.background_gradient(cmap = "Blues")


# ##### Weighted term frequency

# In[22]:


def get_weighted_term_freq(x):
    try:
        return math.log(x)+1
    except ValueError:
        return 0


# In[23]:


wtf = tf.copy()


# In[24]:


for i in range(1,len(dicts)+1):
    wtf[f"DOC_{i}"] = wtf[f"DOC_{i}"].apply(get_weighted_term_freq)
wtf.astype(int).style.background_gradient(cmap = "Blues")


# ### b) Inverse document frequency

# In[25]:


get_term_freq(dicts[1]).keys() 


# In[26]:


idf_df = pd.DataFrame(index = get_term_freq(dicts[1]).keys() , columns=["df" , "idf"])


# In[27]:


for i in idf_df.index:
    idf_df['df'][i] = pos_idx[i][0]
    idf_df['idf'][i] = np.log10( 10 / float(pos_idx[i][0]) )


# In[28]:


idf_df["df"] = idf_df["df"].astype(int)
idf_df["idf"] = idf_df["idf"].astype(float)


# In[57]:


idf_df.style.background_gradient(cmap = "Blues" , axis= 0)


# ### c) Term frequency - Inverse document frequency

# In[30]:


tf_idf = tf.multiply(idf_df["idf"] , axis = 0)


# In[55]:


tf_idf.style.background_gradient(cmap = "Blues" , axis= 0)


# ### Document length & Normalized tf_idf

# ##### Document length

# In[32]:


doc_length = pd.DataFrame()


# In[33]:


def get_doc_length(col):
    return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

for col in tf_idf.columns:
    doc_length.loc[ 0 , col + '_length'] = get_doc_length(col)


# In[34]:


doc_length = doc_length.T


# In[35]:


doc_length.columns = [''] * len(doc_length.columns)


# In[36]:


doc_length.style.background_gradient(cmap = 'Blues')


# ##### Normalized term freq inverse doc freq

# In[37]:


normalized_tfidf = pd.DataFrame()


# In[38]:


def get_normalized_tf_idf(col, x):
    try:
        return x / doc_length.loc[col + '_length'].values[0]
    except ZeroDivisionError:
        return 0


# In[39]:


for col in tf_idf.columns:
    normalized_tfidf[col] = tf_idf[col].apply(lambda x: get_normalized_tf_idf(col , x))


# In[54]:


normalized_tfidf.style.background_gradient(cmap = 'Blues' , axis= 0)


# ### d) Cosine similarity & Ranking documents

# In[41]:


docs_cos = list(dicts.values()) 


# In[42]:


for i in range(len(docs_cos)):
    docs_cos[i] = ' '.join(docs_cos[i])


# In[43]:


vectorizer = TfidfVectorizer()


# In[44]:


df = vectorizer.fit_transform(docs_cos).T.toarray()


# In[45]:


df = pd.DataFrame(df , index= vectorizer.get_feature_names())


# In[53]:


df.style.background_gradient(cmap = 'Blues' , axis = 0)


# In[47]:


def get_relevant_docs(q, df):
    
    print("query is :", q)
    q = remove_stop_words(q)
    print("query read as :", q)
    print("\n------------------\n"*3)
    
    
    q = [q]
    q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0])
    sim = {}
    for i in range(10):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    
    
    print("Most relevant documents:\n")
    returned_docs = []
    print("Cosine similarity: (Query , Document number) ->(Score) : \n")
    for k, v in sim_sorted:
        if v != 0.0:
            returned_docs.append(k+1)
            print(f"({' '.join(q)} , {k+1}) -> {v}")
            print(f"Document: {' '.join(dicts[k+1])}" , end='\n\n')
    print("Returned documents: " , returned_docs)
    
    
    arr_q = []
    for i in ' '.join(q).split():
        try:
            tf_q = sum(np.array(tf.loc[i]))
            wtf_q = get_weighted_term_freq(tf_q)
            idf_q = np.log10( 10 / float(pos_idx[i][0]) )
            tf_idf_q = tf_q * idf_q
            norm_q = tf_q * idf_q / sum(list(doc_length.iloc[[i for i in returned_docs] , 0]))
        except:  
            tf_q , wtf_q , idf_q , tf_idf_q , norm_q = 0,0,0,0,0
        arr_q.append([tf_q , wtf_q , idf_q , tf_idf_q , norm_q])
        
            
    return arr_q


# ## Search engine

# In[48]:


query = input()


# In[49]:


search = get_relevant_docs(query, df)


# # ----------------

# In[50]:


search = pd.DataFrame(search , index= remove_stop_words(query).split() , columns=['tf-raw' , 'wtf(1+ log tf)', 'idf', 'tf*idf' , 'normalized'])


# In[52]:


search.style.background_gradient(cmap = "Reds" , axis= 1)


# In[ ]:





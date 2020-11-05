# 5分钟NLP应用 -文本分类

[toc]



### 1、文本分类-垃圾邮件

#### 1.1、提取tfidf 特征 TfidfVectorizer类

将原始文档的集合转化为tf-idf特性的矩阵。

```
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
```

#### 1.2、分类

利用sklearn 的分类器，   RegressionMnb MultinomialNB，svm(SGDClassifier)，lr=LogisticRegression

定义分类器


    from sklearn.naive_bayes import MultinomialNB   多项式贝叶斯分类器multinomialNB
    from sklearn.linear_model import SGDClassifier  SVM 
    from sklearn.linear_model import Logistic       线性回归     
    
    Regressionmnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter=100)
    lr = LogisticRegression()  


 训练与预测 
    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb,
                                                       train_features=bow_train_features,
                                                       train_labels=train_labels,
                                                       test_features=bow_test_features,
                                                       test_labels=test_labels)
    
    classifier.fit(train_features, train_labels)  # 训练分类器
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
### 2、文本聚类-豆瓣图书聚类

#### 2.1 、 提取tfidf 特征，形成 内容-词 矩阵

```
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
```

#### 2.2 KMeans聚类

```
km = KMeans(n_clusters=10, # 分10类
                max_iter=10000)
km.fit(feature_matrix)             # 聚类
clusters = km.labels_              # 分为 哪一类  5 9 1 1 
book_data['Cluster'] = clusters
```

### 3、情感分析

#### 3.1 基本原理

情感分析本质上是文本分类，一般有如下三种方法

1、关键词判断分析
     词的包含/不包含/，否定词的包含不包含 

2、机器学习  监督学习
      分类特征：unigram，bigrams , trigrams 
                  积极词汇的数量，文档长度，算法SVM , NB, CNN 
3、混合分析
      机器学习(准确)+关键词判断
      用词法分析快速分类文档，然后对文档进行机器学习，例如bayes分类器，Polling 多项式分类器， 在训练中融入手动标记数据。

4、深度学习  

CNN,lstm 等等

#### 3.2 一个LSTM 情感分析示例

1、数据准备

主要是为文本生成词向量索引矩阵ids  25000x300

```
1、载入400000词典  ['and','if']
wordsList = np.load('wordsList.npy')
2、载入400000x50维词向量, [[0,...,1],[]...]  
wordVectors = np.load('wordVectors.npy') 
3、读入文件中的每一行, 计数 
num_words_each_line = [128,242....,...] 
4、统计或者用metaploit 画图统计出最佳的字符序列长度。  
matplotlib.pyplot.hist(num_words,50,facecolor='g')
matplotlib.pyplot.show
5、大部分文本在230内，那么就设为max_seq_len = 300  
得到行数为25000行

6、为文本生成词向量索引矩阵 25000x300 
   ids[fn][i] = wordList[word]     # 例如 中国 1 美国2 


```

2、模型准备  

```
1、设置分类参数
      batch_size=24 
      lstm_unit = 64   ?    num_units这个参数的大小就是LSTM输出结果的维度
      num_labels =2 
      iters=20000 
      lr = 0.001 学习率
      max_seq_len  = 250
      num_dimensions  = 300
2、取一批次的数据
get_train_batch():
get_test_batch():     
      for i in range(batch_size):
              num= randint(25000)
              labels.append([1,0]) 或者 0,1 
              arr[i] = ids[num-1:num]
       return arr,labels  
```

3、定义模型图

```python
import tensorflow as tf


tf.reset_default_graph 
1、定义输入数据
input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])  # 输入数组 从ids 里面取值     [24,250]
labels = tf.placeholder(tf.float32, [batch_size, num_labels]) # 输出数组  [24,2]                     
2、根据输入建构输入的词向量矩阵，作为lstm网络的输入 24X250X50 
data= tf.Variable(tf.zeros([]batchsize,max_seq_len,num_dimensions ),dtype=tf.float)   [24,300,50]
data = tf.nn.embedding_lookup(wordVectors,input_data ) # 24x250x50

3、定义网络
lstm 基本单元
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)  # 输出64维 
定义lstm dropout 
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)

基于lstm cell 构建rnn网络
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
# value [250，24,64]
定义weight 矩阵
weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels])) # [64,2]
定义bais 偏置
bias = tf.Variable(tf.constant(0.1, shape=[num_labels])) # [2,0]
value = tf.transpose(value, [1, 0, 2])  # [250,24,64]    转置value 
last = tf.gather(value, int(value.get_shape()[0]) - 1)  [24,64]  从values的axis维根据indices的参数值获取切片, 去最后一个隐藏状态

计算结果
prediction = (tf.matmul(last, weight) + bias)  # ##  wx+b   [24,2]
损失
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))  
#tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引。 axis 取行中最大值的索引。    
#shape=(24,)  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # 
得到正确率
标准的交叉熵损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)  
```

4、运行

tf.Session.run()

3、调参	


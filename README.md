

# 5minNLP-五分钟了解自然语言处理

看了不少的NLP 的书籍，希望让大家在最短的时间了解NLP. 



## 1、分词

要处理自然语言，首先就要把文本数字化。文本数字化最简单的就是按单个词，或者中文字来数字化。为了包含字的语义信息，一般都是按照词来数字化。那么对于中文来说，碰到的第一个问题就是分词。

以下是常用的分词方法

### 1.1 基于固定词库进行分词

基于词库分词法，一般有 正向最大匹配，反向最大匹配，双向最大匹配3种方法。

正向最大匹配，就是拿文本从左往右，从最长的词开始匹配，找到一个最长的词，就往后移，接着往后找。

反向最大匹配，就是拿文本从右往左找，从最长的词开始找。

双向最大匹配，就是正向一遍，反向一遍，然后，哪个切词数少，就用哪个。

```
词典可以采用有向无环图，始于第一个字，终于最后一个字，和jieba 一样。
```





### 1.2 基于统计的ngram 分词

对文本需要建立一个统计数学模型。

### 1.2.1 ngram 模型

ngram 分词基于n元概率模型， 就是用概率论的专业术语描述语言模型：

为长度为m的字符串确定其概率分布  P(W1,W2,..Wn) = P(W1)P(W2|W1)...P(wn|w1,w2,....Wn-1)

``` 
例如：一句话： 我爱你  .这句话在文本中的概率分布为，就是  "我"的概率 x "我爱"的概率x "我爱你"的概率
```

从简化计算角度出发，一般我们采用二元模型 2-gram

对于二元模型中的，则是：P(W2|W1) = count(W1,W2)/count(W1)，count 就是计数。

```
一个简单的2-gram 分词 可以按照两个词出现的概率值为基准进行分词, 例如计数值>5 认为就是一个词。
```



### 1.2.2 HMM 隐含马尔可夫模型

HMM 是将分词作为字在字串中的序列标注任务来实现的。

他把句子中的字逐字按词标注为Tag：B词首 M词中 E词尾 S单词 。

```
例如：中B文E 分B词E 是S 文B本M处M理E 不B可M或M缺E 的S 一B步E !S
```

Tag 表示标签，那么 最理想的输出为max = max(P(T1T2..Tn|W1...Wn)) 其中：P(T|W) 为W 打上TAG 的条件概率。

```
马尔可夫假设的原理是根据大量的数据分析后，资料、句子的统计，认为：只需根据资料中一个字出现后下一个字紧跟的比例来判断一个句子是否有意义，是否合理，每个输出仅仅与上一个输出有关。得到
最大化P(T|W) ~=  P(W|T)P(T)  ~=  P(W1|T1)P(T2|T1)........P(Tn|Tn-1)P(Wn|Tn) 

P(W1|T1) 为发射概率emit_p，意思是w1 是tag1 时的概率。
P(T2|T1) 为转移概率trans_p，意思是T1后面是T2的概率。
```

#### 发射概率P(W1|T1)的计算

```c++
出现一个词，那么概率就加1 就是发射概率。 
例如：abc - BME ,   P('a'|'B') = {'B':{'a':1}}
最终结果示例如下：

B：<class 'dict'>: {'中': 12812.0, '儿': 464.0, '踏': 62.0, '全': 7279.0, '各': 4884.0,
M：<class 'dict'>: {'９': 6418.0, '８': 4997.0, 
E：<class 'dict'>: {'年': 15983.0, '亿': 695.0, 
S：<class 'dict'>: {'，': 193584.0, '新': 4865.0, '的': 142506.0, 
                     
转换成百分比:
B  <class 'dict'>: {'中': 0.009227731157798309, '儿': 0.00033488605232000413, 
M  <class 'dict'>: {'９': 0.02860542429076908, '８': 0.02227292578365226, 
E  <class 'dict'>: {'年': 0.01151143797910311, '亿': 0.0005012488008918772, 
S  <class 'dict'>: {'，': 0.12024540410804042, '新': 0.0030225179450356415,
```

#### 转移概率P(Tk|Tk-1) trans_p 计算实例

```c++
前一个字的tag 为 T1,  后一个字为T2 ,  那么 当 T1T2 出现时， p[T1,T2] = p[T1,T2]+1 
另外需要排除BBB,EM等不合理组合
转移概率 trans_p
{'B': {'B': 0.0, 'M': 162066.0, 'E': 1226466.0, 'S': 0.0}, 
 'M': {'B': 0.0, 'M': 62332.0, 'E': 162066.0, 'S': 0.0}, 
 'E': {'B': 651128.0, 'M': 0.0, 'E': 0.0, 'S': 737404.0}, 
 'S': {'B': 563988.0, 'M': 0.0, 'E': 0.0, 'S': 747969.0}}
统计百分比
B:{'B': 0.0, 'M': 0.1167175117318146, 'E': 0.8832824882681853, 'S': 0.0}
M:{'B': 0.0, 'M': 0.2777743117140081, 'E': 0.7222256882859919, 'S': 0.0}
E:{'B': 0.46893265693552616, 'M': 0.0, 'E': 0.0, 'S': 0.5310673430644739}
S: {'B': 0.3503213832274479, 'M': 0.0, 'E': 0.0, 'S': 0.46460125869921165}

```

#### 状态初始概率

 ```
<class 'dict'>: {'B': 173416.0, 'M': 0, 'E': .0, 'S': 124543.0}
也要统计百分比
 ```

统计完上述概率之后，模型建立完成。



#### HMM分词-采用veterbi 算法


上面的统计概率模型，采用viterbi 维特比算法，到达每一列的时候都会删除不符合最短路径要求的路径，大大降低时间复杂度。
每次取一个字，计算起始状态的每条路径的路径概率,去每条路径的最大的路径。
这样，每次处理一个字，保留的都是最多4条路径（最多4个状态）

```python
示例代码
V= start_p[y] * emit_p[y].get(text[0], 0)
#<class 'list'>: [{'B': 0.003291232115235236, 'M': 0.0, 'E': 0.0, 'S': 0.0012044407157278893}]
path = <class 'dict'>: {'B': ['B'], 'M': ['M'], 'E': ['E'], 'S': ['S']}

for t in range(1, len(text)):  #从第二个字开始循环计算
            V.append({})
            newpath = {}
            for y in states:
                emitP = emit_p[y].get(text[t], 0)  #取当前字的发射概率
                (prob, state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0)  *emitP, y0) for y0 in states)  # 针对每个状态取得当前最大的路径
                V[t][y] = prob        
                newpath[y] = path[state] + [y]  # 确定一条路径
            path = newpath   
      (prob, state) = max([(V[len(text) - 1][y], y) for y in states]) #取最后的最大值
      return (prob, path[state])  #返回最大的路径
```

jieba 实现了hmm 分词算法。 


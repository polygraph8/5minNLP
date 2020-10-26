# 五分钟NLP - 句法分析

## 一、理论基础：Chomsky形式文法

Chomsky文法用 G G*G* 表示形式语法，将其表示为四元组：
**G = ( Vn , Vt , S , P )**

*Vn*：非终结符的有限集合，即在实际句子中不出现。相当于语言中的语法范畴；例如名词N

*Vt*：终结符的有限集合，只处于生成过程的终点，是句子中实际出现的符号，相当于单词表。

*S*：Vn中的初始符号，相当于语法范畴中的句子。

*P*：重写规则，也成为生成规则，一般形式为α → β ，其中α 、 β 都是符号串，α至少含有一个Vn中的符号。

语法 G的不含非终结符的句子形式称为 G生成的句子；
由语法 G生成的语言，记做 L(G)，指 G 生成的所有句子的集合。

例如：

```
假设有一种语言 { a b , a a b , a a a b , a a a a b . . . } \{ab,aab,aaab,aaaab...\}
Vn={S,A}
Vt={a,b}
P: S→Ab, A→aA∣a
当然也可以表示成： P:S→aA,A→aA,A→b
```

**0型文法（无约束文法）**：重写规则为 α → β ，其中，α , β ∈ ( V n ∪ V t ) 

**1型文法（上下文相关文法）**：重写规则为αAβ → αγβ，其中A∈ Vn,α, β,γ ∈( V n∪V t ) ∗ ，*γ* 非空

**2型文法（上下文无关文法CFG）**：重写规则为 A→ α，其中，A∈V n , α ∈( V n∪V t) , A重写为 α时没有上下文限制。【树形结构从上往下，叶子节点为句子】

**3型文法（正则文法RG）**：重写规则为 A→Bx或 A→x，其中，A,B∈ Vn , x∈V t 【由状态 A转入状态 B时，可生成一个终结符x】

```
Chomsky范式 :
Chomsky证明，任何由上下文无关文法生成的语言，均可由重写规则为A → B C或者 A → x的文法生成，其中A , B , C ∈ Vn , x ∈ Vt 
```

由此文法生成句子的过程图如下：

![image-20201018225934584](C:%5CUsers%5CAdministrator%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201018225934584.png)



## 二、规则句法分析

**自底向上的方法**从句子中的词语出发，基本操作是将一个符号序列匹配归约为其产生式的左部（用每条产生式左边的符号来改写右边的符号），逐渐减少符号序列直到只剩下开始符S为止。
**自顶向下的方法**从符号S开始搜索，用每条产生式右边的符号来改写左边的符号，然后通过不同的方式搜索并改写非终结符，直到生成了输入的句子或者遍历了所有可能的句子为止。



**线图（chart）分析算法**

线图法将每个词看作一个结点，通过在结点间连边的方式进行分析，查看任意相邻几条边上的词性串是否与某条重写规则的**右部**相同，如果相同，则增加一条新的边跨越原来相应的边，新增加边上的标记为这条重写规则的头（左部）。重复这个过程，直到没有新的边产生。

<img src="https://img-blog.csdnimg.cn/20191230194325108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzc4MjIx,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 50%;" />



**CYK方法**

CYK算法是自底向上的句法分析方法，通过构造识别矩阵进行分析，时间复杂度相对线图法有所减小。

<img src="https://img-blog.csdnimg.cn/20191231223810384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5Mzc4MjIx,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 33%;" />



## 三、概率统计句法分析算法**PCFG**

统计语言学中词与词、词与词组以及词组与词组之间的规约信息，并且可以由语法规则生成给定句子的概率。

主要作用：

1.句法分析树的消歧
2.求最佳分析树

一个PCFG由如下五个部分组成 ： 

 ( Vn , Vt , S , R(生成规则) )+    P(R) :  对任意产生式 r ∈ R，其概率为 P ( r ) ，

 或者说产生规则都是带概率的。  

<img src="https://img-blog.csdnimg.cn/20190307181609935.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VjaG9LYW5nWUw=,size_16,color_FFFFFF,t_70" alt="PCFG" style="zoom:50%;" />

有了概率，消歧就很容易了：

![img](https://img-blog.csdnimg.cn/20190308093821571.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VjaG9LYW5nWUw=,size_16,color_FFFFFF,t_70)

## 四、找出最佳语法树 **PCFG Viterbi算法**

输入是文法 G ( S ) 以及语句 *W*=*w*1*w*2...*wn*

*γ**i**j*(*A*)：非终结符 A推导出语句 W W*W* 中子字串 w i w i + 1 . . . w j{i,j}

*ψi*,*j*：记忆字串 *w**i**wi+1...wj 的Viterbi语法分析结果

![image-20201018234218185](C:%5CUsers%5CAdministrator%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20201018234218185.png)

从下往上逐层递归。



例如：

![PCFG Viterbi](https://img-blog.csdnimg.cn/20190322101807614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VjaG9LYW5nWUw=,size_16,color_FFFFFF,t_70)

![结果示例](https://img-blog.csdnimg.cn/20190322102229397.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VjaG9LYW5nWUw=,size_16,color_FFFFFF,t_70)


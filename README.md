# FL project

这是SJTU AI3602 Data Mining 2021学年的课程大作业，我们小组的选题是与联邦学习（Federated Learning）有关的内容。

## 主要工作

* 复现了FL领域中著名的FedAvg算法，且应用在我们的lab2编程作业node2vec上
* 添加了FedAvg的并发支持，使用python的Threading库
* 观察到client drift现象，作为解决方案我们调研了SCAFFOLD算法并且应用在node2vec上
* 训练时试图采取权重混合的方式提高模型面对先前未选中的client时的收敛速率，但是对跑分有影响所以最后没有加。我们的代码仍然是支持这个任务的

## 性能跑分

写在前面，由于选择的模型过于简单（node2vec跟deep learning有关的结构就两层embedding），很容易过拟合，所以其实不用FL更好。
有人可能会问为什么不换模型，当我们发现这个问题的时候已经快到ddl了，课程论文都写了大半了，来不及换...
所以性能跑分这部分很遗憾我们不能给出很好的结果，我们的主要还是重在尝试和发现问题。

| method | AUC score (%) |
| ------ | ------------- |
| baseline (lab2) | 95.84 |
| FedAvg | 84.44 |
| SCAFFOLD | 84.09 |

复现方式：以下三条命令行分别对应上面的三个输出
```shell
python main.py -m o -b 5000 -lr 0.01
python main.py -s 2021
python main.py -s 2021 -m s
```
随机数种子为2021。其他参数的性能跑分及其复现代码见下文。

## 运行代码

完整的参数列表为（在根目录下通过控制行运行）

```shell
python main.py -m [训练模式] -s [随机数种子] -b [batch size] -lr [学习率]
-p [出入概率] -q [返回概率] -ew [随机游走的轮数] -e [普通训练的轮数] -dim [词嵌入维度]
-es [全局轮数] -ec [每个client的轮数] -c [client个数] -r [client被选中的概率]
-t [多线程训练] -buf [使用历史权重记录] -rmix [权重混合的比例]
-is [每个client数据集大小相同] -i [每个client数据集iid]
```

### 生成随机游走路径

推荐使用
```shell
python main.py -m w
```
* `-m`：本次实验采取的训练模式，一共有四个备选值。`w`代表生成随机游走链，`o`代表常规训练（同lab2），`f`代表使用FedAvg训练，`s`代表使用SCAFFOLD训练
* `-p`和`-q`：node2vec采样获得随机游走链的出入概率`p`和返回概率`q`，默认1
* `-ew`：生成随机游走链的epoch数，以每个节点为出发点生成多少条链，默认100

### 普通训练（非FL框架）

推荐使用
```shell
python main.py -m o
```
* `-e`：训练的epoch数，默认值1
* `-dim`：word embedding的词嵌入向量维度，默认100
* `-b`：batch size，默认10000
* `-lr`：学习率，默认0.025

### FedAvg

推荐使用
```shell
python main.py
```
* `-es`：global epoch，一共需要通信多少次（每一次通信，server完成权重聚合并下发给每个client进行新一轮训练），默认5
* `-ec`：local epoch，每个client自己训练多少轮之后与server进行通信，默认5
* `-c`：client的数量，默认10
* `-r`：每个client被选中的概率，默认0.4
* `-buf`：是否启用权重历史记录进行训练，默认False
* `-rmix`：如果启用权重历史记录，以多大的比例混合过往权重与当前权重，默认0.5

### FedAvg并发（多线程）

推荐使用
```shell
python main.py -t 1
```
* `-t`：在FedAvg中是否开启多线程训练，默认0（False）。如果开启，每个client会被分配一个线程并发训练

### SCAFFOLD

推荐使用
```shell
python main.py -m s
```
它没有特殊参数，与FedAvg共享参数

## 可复现性

依照下面的控制行代码，可以基本复现出论文里各表格展示的结果。为了保证可复现性，所有结果均来自随机种子训练（seed training）。

### Tab II

| method | AUC score (%) |
| ------ | ------------- |
| baseline (lab2) | 95.84 |
| FedAvg | 84.44 |
| SCAFFOLD | 84.09 |

```shell
python main.py -m o -b 5000 -lr 0.01
python main.py -s 2021
python main.py -s 2021 -m s
```

### Tab III

| method                | AUC score (%) |
|-----------------------|---------------|
| FedAvg                | 72.29         |
| FedAvg (multi-thread) | 72.61         |
| SCAFFOLD              | 73.61         |

```shell
python main.py -s 2022 -es 10 -ec 10
python main.py -s 2022 -es 10 -ec 10 -t 1
python main.py -s 2022 -es 10 -ec 10 -m s 
```

### Tab IV

| method           | AUC score (%) |
|------------------|---------------|
| FedAvg           | 84.44         |
| FedAvg (mix=0.1) | 84.04         |
| FedAvg (mix=0.5) | 77.00         |

```shell
python main.py -s 2021
python main.py -s 2021 -buf 1 -rmix 0.1
python main.py -s 2021 -buf 1 -rmix 0.5
```

### Tab V

| method    | AUC score (%) |
|-----------|---------------|
| seed=2021 | 84.94         |
| seed=2022 | 55.29         |

```shell
python main.py -s 2021
python main.py -s 2022
```

### Tab VI

| training set size | AUC score (%) |
|-------------------|---------------|
| Identical         | 84.94         |
| Non-identical     | 78.36       |

```shell
python main.py -s 2021
python main.py -s 2021 -is 0
```

### Tab VII

| training set distribution | AUC score (%) |
|---------------------------|---------------|
| iid                       | 84.94         |
| Non-iid                   | 58.71         |

```shell
python main.py -s 2021
python main.py -s 2021 -i 0
```


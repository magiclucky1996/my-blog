---
title: week8&9
date: 2023-05-05 10:19:40
tags: work
---

# week 8&9

### Goal(这周的目标)

- rl核心
  - **基于梯度下降的内核**
  - **q表怎么扫，环境中什么轨迹，怎么采样，怎么更新**
  - **神经网络本身特性带来的一些东西**
  - **神经网络的作用就是状态和动作空间太大，那我们给他缩小，最后我们用缩小的q表去采取动作（generalize function）**
    - generalize func的性能没有NN好
  - 而pg只是基于价值大小提高动作概率，只是argmax的一种连续版本
  
- 群体学习的解决方式（我的目的不是发论文，而是探索，mastering）

- 随机性的问题，也就是探索的问题，探索环境状态的问题

- playing atari中的提到的神经网络的优点
- huggingface中提到的深度神经网络的优点
  - q表在大型空间中无效（为啥，可以缩小，模糊神经网络）
- rlbook中提到神经网络的优点
- 把搜索方法应用到对mdp空间的探索中，比如树搜索，比如rrt，比如采样方法
  - 我们基于简单的mdp找出最佳的探索方法，再应用到复杂的，无法画出关系图的mdp中
  - ![image-20230508115549166](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230508115549166.png)







|                                                              | 星期一 | 星期二 | 星期三 | 星期四 |
| :----------------------------------------------------------- | ------ | ------ | ------ | ------ |
| 抱脸虫教程（往后放，先解决问题）                             |        |        |        |        |
| 在env测试不同算法进行对比                                    |        |        |        |        |
| 做一个图对比各种采样的策略：softmax，epsilon greedy，fully explore |        |        |        |        |
| 做一个图对比各种对模型的更新                                 |        |        |        |        |
| 对比使用神经网路和不使用神经网络（缩小的q表，q也可以根据价值选取动作概率，，所以为什么使用神经网络） |        |        |        |        |

### Q1： why neural network？

###### q table could be approximate, actor could be like argmax

- rlbook
  - Generalize problems has already been studied
  - generalization: when a single state is updated, the change generalizes from that state to a↵ect the values of many other states
  - supervised learning from examples, including artiﬁcial neural networks, decision trees, and various kinds of multivariate regression
  - not all function approximation methods well suited for rl
    - If online, needed to learn efficiently from incrementally acquired data\
    - target is nonstationery（policy is not stationery, td is not stationery）
    - 



- DQN paper





- chatgpt





- nn could represent more complex relationships: so it's more complex than just 在q 表中取近似值



### Q2: the explore of  state-action space: 

###### why actor used to sample(softmax, epsilon greedy, fully explore), contraction between them





### Q3: the explore of  state-action space: 

###### Can search methods be applied? like A*, RRT





#### Q4: the explore of state-action space: 

###### how to scan the q table?





### Questions:

1. why q table replaced by NN, what is the discipline to represent? （q表换成q神经网络的意义是什么）

2. difference b
<<<<<<< HEAD
3. fully explore 和 epsilon greedy
4. why actor better than argmax
=======

3. fully explore 和 epsilon greedy（一开始完全探索）

4. why actor better than argmax（actor网络和argmax1的区别是什么）

5. 神经网络本身的特性会带来哪些东西，神经网络的优点是什么

6. q learning能否动作是概率分布，为什么不行?

   1. 可以，使用softmax作为探索的策略，**和epsilon greedy的区别**：会按照q值选取动作，不能保证对动作空间的充分探索
   2. 在探索时，选取最优动作，和选取没选取过动作的区别：
   3. **探索的衡量指标**：做一个图对比各种采样的策略

   


>>>>>>> 9c004c5 (modify week 8)



### tips

- fully explore（完全探索）

This approach is called pure exploration and exploitation (PEE) and can be used in some cases where exploration is very costly or where the environment is very simple. However, in most real-world scenarios, PEE is not an optimal approach.

The problem with PEE is that the agent spends a lot of time exploring and collecting data, but not enough time exploiting that data to improve its policy. This can lead to slow learning and poor performance, especially in complex environments where there are many possible actions and states.

In contrast, most RL algorithms use a balance between exploration and exploitation, where the agent takes actions that are likely to yield high rewards based on its current policy while also occasionally exploring new actions or states. This allows the agent to learn quickly while still exploring new possibilities, leading to faster learning and better performance.

Furthermore, in many RL problems, the environment is dynamic and can change over time. In such cases, it is important for the agent to continuously explore and adapt to changes in the environment to maintain optimal performance. This requires a balance between exploration and exploitation, as well as the ability to update the policy based on new data and experiences.

In summary, while PEE can be a useful approach in some cases, a balanced approach between exploration and exploitation is generally more effective for most RL problems.



我用fully explore探索，然后得到的数据存在buffer里，然后用我的策略进行学习，差策略探测到的数据和提升后的策略探索到的数据的区别是什么，数据是一个transition（s a r s ）好的策略往往会偏向于采取好的动作，转移到好的动作

- 好的策略会选好的动作，到好的效率上，efficiency会很高，比如我一直用随机策略采样，然后用
- 坏的策略会选坏的动作，到不好的轨迹上，

所以现在两个最核心的问题：

1. Fully 随机策略就只是影响轨迹，进而影响采样效率吗
2. 看看rlbook里资格迹的概念
3. 看看里面q表怎么扫的
4. 基于q的方法和基于policy的方法本质是一样的，都是value iteration，通过r来提升value，最后通过value去驱动actor，其实本质仍然是用r去驱动一切，核心就在于r怎么去提升策略，即每个状态做什么动作，如何用r去帮助这个状态做什么动作，本质就是看看哪个动作得到的r多，一种是累积r，一种是最终r，现实世界可能都是最终r，但是这样很难学习，所以我们一般先给累积r，然后最后通过最终r去实现，，，

所以在这个状态选哪个好呢，一种是sample一下，一种是sample到尽头，sample到一下，sample到一步的r，那我必须bootstrap，来得到当前状态选取某个动作的value，也可以sample轨迹，几个轨迹平均一下，看看动作的value，都是计算动作的价值，如果是连续动作空间怎么办呢，，，，那其实两个在参数上相近的（s，a）对应的价值可能差的很大


<<<<<<< HEAD
# Incoporate  mappo with sumo
=======
>>>>>>> 9c004c5 (modify week 8)















## incoporate  mappo with sumo（代码部分：在sumo中实验信号灯控制）

### 1. look for existing mappo + sumo

- reference

1. github
2. paper
3. resource of course era, presentation, tutorial (osint)

- resources found

github

1. ppo + sumo https://github.com/maxbren/Multi-Agent-Distributed-PPO-Traffc-light-control

2. light mappo  https://github.com/magiclucky1996/light_mappo 基于这个写一下试试
3. q/ac + sumo https://github.com/magiclucky1996/deeprl_signal_control

4. mappo + mujoco https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation
5. mappo/qmix/maddpg + mpe https://github.com/Lizhi-sjtu/MARL-code-pytorch
6. ppo + sumo https://github.com/YanivHacker/RLTrafficManager
7. noisy mappo+  https://github.com/hijkzzz/noisy-mappo

papers

https://ieeexplore.ieee.org/abstract/document/9549970/authors#authors 东北信息学院：提到了rl奖励稀疏的问题，然后他们给rl设计更多的reward 引导它学习。但是你怎么知道什么样的reward能够引导，这不是还是在reward function设计的范围里吗？





- insights
  1. sumo设计的问题，如果我们设一个控制周期之后的交通流状态为reward是不是不合理，怎么样去评价智能体schedule的好坏呢，怎么去评价智能体的action改善了交通呢，我是要搞交通呢，还是要搞rl呢，还是要搞啥，
  2. 上次会议的要点：1. insight可以给硕士做，但是要具体可行 2. 做一个oncoming 会议的scheduling(rl 多智能体 交通 计算机 人工智能) 3. sumo的模型可以封装好给本科生用 4.  
- 今天的计划（4.24）

work 到12 点半，吃中饭，吃完中饭一点消化一会儿，回实验室一点半，回来继续work,work到两点多的时候睡午觉，五点跑路，去看看有没有吃的，不行就回家supervalu,晚上回家继续work一会儿，今天的弄完走之前deploy 和 push上去







- how other people implement marl training





make contraction between these projects

分解成detailed steps

1.首先看下mappo的代码和deeprl sumo的代码以及ppo sumo代码（只做有必要做的事情）

2. 做完1大概知道要干嘛，可以看看ppo trpo mappo

3. 把每周4上午留作整理时间，所以我必须两天内搞定这个代码整定的事情，然后再用剩下的时间学习ppo trpo mappo，还要学sumo部分的东西，但是得非常快

4. 

5. 

   



**reading of the rlbook**


<<<<<<< HEAD

how does dopamine give people motivation?

if prediction is different from truth, we will have positive feedback or negative feedback?

## 
=======
### 2. make it with materials found
>>>>>>> 9c004c5 (modify week 8)





**生活的star：**

piano

上海美术厂

breaking

penetration testing

avoid social media



## note

### 抱脸虫教程

https://huggingface.co/learn/deep-rl-course/unit1/exp-exp-tradeoff?fw=pt

- explore/exploit权衡：本质是什么：如果exploit，就会一直选择目前的最优解，但是目前的最优解怎么得来，初始化的时候怎么选择，，所以用epsilong decay的epsilon greedy
- 如果我一开始完全探索，像value表格一样，如何扫过去（也就是如何更新q值），看看rl book，
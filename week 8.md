---
title: week8&9
date: 2023-05-05 10:19:40
tags: work
---

# week 8

### goal of the week

reading rl papers, marl papers, sutton rlbook2020





### Questions:

1. why q table replaced by NN, what is the discipline to represent?
2. difference b
3. fully explore 和 epsilon greedy
4. why actor better than argmax
5. 





### tips

- fully explore

This approach is called pure exploration and exploitation (PEE) and can be used in some cases where exploration is very costly or where the environment is very simple. However, in most real-world scenarios, PEE is not an optimal approach.

The problem with PEE is that the agent spends a lot of time exploring and collecting data, but not enough time exploiting that data to improve its policy. This can lead to slow learning and poor performance, especially in complex environments where there are many possible actions and states.

In contrast, most RL algorithms use a balance between exploration and exploitation, where the agent takes actions that are likely to yield high rewards based on its current policy while also occasionally exploring new actions or states. This allows the agent to learn quickly while still exploring new possibilities, leading to faster learning and better performance.

Furthermore, in many RL problems, the environment is dynamic and can change over time. In such cases, it is important for the agent to continuously explore and adapt to changes in the environment to maintain optimal performance. This requires a balance between exploration and exploitation, as well as the ability to update the policy based on new data and experiences.

In summary, while PEE can be a useful approach in some cases, a balanced approach between exploration and exploitation is generally more effective for most RL problems.





# incoporate  mappo with sumo

## 1. look for existing mappo + sumo

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

   





## 2. make it with materials found















生活的star：

piano

上海美术厂

breaking

penetration testing

avoid social media
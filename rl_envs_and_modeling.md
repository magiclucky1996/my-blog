---
title: rl_play
date: 2023-05-05 10:19:40
tags: work
---

# play with gym env

## cart pole

control + pid: https://blog.csdn.net/weixin_44562141/article/details/119700574

document of gym https://www.gymlibrary.dev/environments/classic_control/cart_pole/

driving test http://www.theory-tester.com/questions/358

- state space:

position

velocity

angle

angular velocity

- first understand problem, then understand reinforcement learning, u must understand the env, then you know why their study is like that, try to be a good teacher
- for spare time, can play , for working time, only do things creating value to this project.
  -  first look for mappo implementation, then try doing it by myself

- if we want to control it with pid,
-  in this env, u are just study "action 要和夹角反着"+ 夹角和几个输入数据的关系，一部分是先验只是可以给的，所以我先原始地学习一下，再把state加工一下加进去，再试试一下把控制的东西加进去，对，我得先有想法，再实验，再读文献，再自己思考，再实验。我希望按照自己的想法来，这样我会沉迷于探索。我希望一直自己保有一些探索的时间，最后发现科研的乐趣。

## frozen lake

https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit2/unit2.ipynb#scrollTo=Y1tWn0tycWZ1

```python
qtable = np.zeros(state_space, action_space)
action = argmax(qtable[state][:])
```

```python
# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7          # Learning rate(这个不是梯度下降的learning rate,是td error的learning rate 

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode（防止回合死循环）
gamma = 0.95                 # Discounting rate （value的discounting）
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters （刚开始探索大，后来探索小）
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability 
decay_rate = 0.0005            # Exponential decay rate for exploration prob
```





## lunarLand

安装

pip install box2d-py





## taxi

https://huggingface.co/learn/deep-rl-course/unit3/introduction?fw=pt









## github ball game







## hugging face tutorial







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





生活的star：

piano

上海美术厂

breaking

penetration testing

avoid social media
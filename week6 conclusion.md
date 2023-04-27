# week6 conclusion

# 1. Trpo PPO MAPPO

# 1.1 Trpo

**(1). Approximation**

- the cost function

![image-20230426101337008](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426101337008.png)

- the V

![image-20230426101210597](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426101210597.png)

- so it could be approximated by

![image-20230426105145352](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426105145352.png)

- therefore J could be transferred to 

  ![image-20230426101435439](/home/sky/.config/Typora/typora-user-images/image-20230426101435439.png)



- try to find the **theta** which could maximize the **J**
- **S** follows the trajectory of steps, but are seen as stochastic sampling from the env
- A is also sampled by the strategy Pi

- collect this trajectory by interacting with the env
- then it would be like this

![image-20230426104602682](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426104602682.png)



#### (2). optimization

- in trust region, update parameter, 

![image-20230426105534532](/home/sky/.config/Typora/typora-user-images/image-20230426105534532.png)

it's an optimization problem, we construct the optimization problem, then throw to optimization solver to solve it.

#### (3). pseudocode

![image-20230426120749975](/home/sky/.config/Typora/typora-user-images/image-20230426120749975.png)





In one cycle, the strategy network is updated each time, and one game is played to obtain a trajectory. However, in maximization, there are multiple inner cycles required by optimization problems , which are usually solved by gradient projection algorithm.



- 2 hyperparameters 4 maximization: 
  - Step size of gradient descent, 
  - radius of confidence region







# 1.2 PPO

- PPO version 1: add constraint into cost function

![image-20230426120941839](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426120941839.png)

![image-20230426120839455](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426120839455.png)



![image-20230426123355644](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426123355644.png)

https://arxiv.org/abs/1707.06347



|      | cons                                                         |
| ---- | ------------------------------------------------------------ |
| DQN  | 1. fails on simple problems; <br />2. poorly understood      |
| VPG  | 1. poor data efficiency <br />2. poor robustness             |
| trpo | 1. complicated <br />2. not compatible with noise ( like dropout)+ data sharing |
| ppo  | 1. good data efficiency<br />2. reliable profermance<br />3. only first -order optimization |
|      |                                                              |







# 1.3 MAPPO



https://jianzhnie.github.io/machine-learning-wiki/#/deep-rl/papers/Overview

centralized critic

![../_images/MARL_cooperation_algo.png](https://raw.githubusercontent.com/magiclucky1996/picgo/main/MARL_cooperation_algo.png)



Valued-based MARL

roma

Qmix

Actor-Critic MARL

maac

coma

maddpg

mappo



# 2. pieter abbeel rl course

https://www.youtube.com/watch?v=2GwBez0D20A&t=130s

## 2.1 : MDP

***insight: group of robot learn faster: data sharing, more efficient sampling of the env***



- groups of robots learn faster, they can share date, more efficient sampling of the env, save wall time
- gamma (discount factor) is also designed based on what our goal is, if we want the agent of care more about things happen in closer steps, then ...
- if gamma is introduced, state take less  steps to the reward is with higher value, it's like the "time" of game world. but it should not be same as the future evaluated in our real world.
- update: in grid world,  we swap a time for all the grid , what if we use different way to swap all the states?
  - like along trajectory
  - like importance sampling
-   Discount factor influence convergence: 0: faster 1: longer
- why it converge

![image-20230426134554214](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426134554214.png)





##### effect of discount and noise







![image-20230426150334001](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426150334001.png)

(a)

![image-20230426171844052](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171844052.png)



(b)

![image-20230426171913044](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171913044.png)



(c)\

![image-20230426171946200](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171946200.png)





(d)

![image-20230426172016159](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426172016159.png)







- update for Q*  , as default,  the agent thereafter acting optimally

![image-20230426172142184](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426172142184.png)





- policy evaluation



### max entropy





- how do we collect data
  - use current policy to collect data	, if policy is deterministic , data collection would not be interesting
  - with entropy, policy will be with more variation in how the data is collected  





- entropy 



![image-20230426173232834](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426173232834.png)









- a distribution over near-optimal solution
  - **robust policy**:  the env could change, if it's distribution instead of deterministic it's more robust
  -  **robust learning** : we can keep collecting exploratory data during learning 

- collect along learning, or collect with exploration , then off-policy update

- insights: how about when the best action changed in a state , increase the   possibility of explore and update to former states,( like in my last paper, in board are , increase explore, in narrow area, reduce explore)
- insights:  after update the value of a state , trace back and update all the former state (read the trace chapter of sutton book)



## 2.2 : Q learning

**properties**

- converge even if act suboptimal (epsilon greedy)
- epsilon decay: if not do decay , latest experience will make you hop around
- epsilon: u need to make it small eventually 
- epsilon: cannot decay too fast: cannot update enough

**requirement**

- state and actions are visited infinitely often: doesn't matter how u select actions
- learning rate schedule: 
  - reference: [On the Convergence of Stochastic Iterative Dynamic ...](https://dspace.mit.edu/bitstream/handle/1721.1/7205/AIM-1441.pdf?sequence=2)



- 



# 3. incorporate MAPPO with sumo









# 4. Interact with rl algorithms



- rl playground

https://rlplaygrounds.com/



- openai gym

- deep mind control 

https://github.com/deepmind/dm_control



- Unity ML-Agents (

 https://github.com/Unity-Technologies/ml-agents )



- course from neptune

https://neptune.ai/blog/best-reinforcement-learning-tutorials-examples-projects-and-courses



- easy game to visualize reinforcement learning

https://towardsdatascience.com/reinforcement-learning-and-visualisation-with-a-simple-game-a1fe725f0509

when state space is large, the update of qtable will be very slow...





*paper to be download:* 

- *playing atari with drl* https://arxiv.org/abs/1312.5602
- *rainbow*  https://arxiv.org/abs/1710.02298
- *ppo* https://arxiv.org/abs/1707.06347
- *mappo* https://arxiv.org/abs/2103.01955
- *maven* https://arxiv.org/abs/1910.07483
- qmix https://arxiv.org/abs/1803.11485
- maddpg https://arxiv.org/abs/1706.02275
- coma https://arxiv.org/abs/1705.08926
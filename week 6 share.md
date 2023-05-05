---
title: week6
date: 2023-05-05 10:19:40
tags: work
---



# week 6&7 share



### Algorithms-single agent(单智能体算法总结)



- **算法1： DQN**
  - valued based (unstable because the update of strategy is not smooth)
  - poorly understood
  - rainbow is the best implementation version of DQN (https://arxiv.org/abs/1710.02298)

![Pseudo-code of DQN with experience-replay method [12]](https://www.researchgate.net/publication/333197086/figure/fig11/AS:941946201727001@1601588883526/Pseudo-code-of-DQN-with-experience-replay-method-12.png)



- **算法2： REINFORCE**

![reinforcement learning - Why the $\gamma^t$ is needed here in REINFORCE:  Monte-Carlo Policy-Gradient Control (episodic) for $\pi_{*}$? - Cross  Validated](https://i.stack.imgur.com/8Jn8l.png)

- **算法3： VPG**
  - poor data efficiency
  - poor robustness

![Vanilla Policy Gradient — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)



- **算法4： DDPG**
  - sampled from replay buffer
  - good data sampling efficiency (why , just because of reuse of data in buffer)
  - what does the deterministic refer to?

![Deep Deterministic Policy Gradient — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg)



- **算法5： SAC**
  - max entropy version of ddpg
  - entropy help to explore

![Soft Actor-Critic — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)

- **算法6： TRPO**
  - not compatible with frame including noise or parameter sharing (ppo paper)
  - a little bit complicated

![Trust Region Policy Optimization — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/5808864ea60ebc3702704717d9f4c3773c90540d.svg)

- **算法7 PPO**

  - data efficiency is not as good as ddpg: why?

  - collect trajectory set, improve value function to be close to utility with multiple gradient descent

    

![Proximal Policy Optimization — Spinning Up documentation](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)



#### chart for comparison对比表格

|      | value and policy                                             | improve method                                               | sampling                                            | update of model                                              | facts                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DQN  | value: **Q network**<br />policy: **argmax Q**<br />sample: **epsilon_greedy** | **Bootstrap**: receive r, improve Q                          | sample along **trajectory** with **epsilon_greedy** | one step interact + experience replay                        | off-policy(replay buffer)<br />discrete action               |
| DDPG | value: **Q network**<br />policy: **actor ** <br />sample: **actor** | **Bootstrap**: receive r, update Q network , imporve         | sample along trajectory with **actor**              | one step interact + experience replay                        | off-policy;(replay buffer)<br />continuous action space<br /> |
| SAC  | value: **Q network**<br />policy: **actor**<br />sample: **actor** | **Boorstrap**: receive r, update Q network, improve **actor** | sample along trajectory with **actor**              | one step interact+experience replay                          | off-policy(buffer)<br />continuous or discrete action space  |
| VPG  | value: **Q network**<br />policy: **actor**<br />sample:**actor** | **MC**: collect set of traj, improve **Actor** with PG, then improve **value** network | sample along trajectory with **actor**              | interact for trajs+ update **actor** with PG(advantage by **critic**); update **critic** with MC | **on-policy**;<br />discrete+continuous action(policy network output could be action)<br /><br />easy to trapped in local optima |
| TRPO | value: **MC**<br />policy: **actor**<br />sample: **actor**  | **MC**: collect set of traj, improve **Actor** with PG, then improve **value** network | sample along trajectory with **actor**              | interact for trajs+ update **actor** with PG(advantage by **critic**); update **critic** with MC | on policy                                                    |
| PPO  | value: **MC**<br />policy: **actor**<br />sample: **actor**  | **MC**: collect set of traj, imporve **actor** with PG, then improve **value** network | sample along trajectory with **actor**              | interact for trajs+ update **actor** with PG(advantage by **critic**); update **critic** with MC | on policy                                                    |
|      |                                                              |                                                              |                                                     |                                                              |                                                              |
|      |                                                              |                                                              |                                                     |                                                              |                                                              |
|      |                                                              |                                                              |                                                     |                                                              |                                                              |



### analysis for rl frame(rl框架的分析)

**Bootstrap**: collect transition , store in the replay buffer, update model every state in the env, **(DQN, DDPG, SAC)**, it's usually **off-policy, good data efficiency**(data in buffer could be reused), the update of model is more frequent( every step in the env)

**MC**: Collect trajs with current policy, improve **actor** with utility through PG, then improve value with utility **(VPG\ TRPO\PPO)**, it's usually **on-policy**, **data efficiency** is worse than **ddpg and sac**, update of model is not that frequent, 





**solving iid problem**: 

- replay buffer: off-policy method, applied for **bootstrap**.

**solving explore problem**: 

- epsilon-greedy: applied for value based rl 
- max entropy: applied for policy based rl





### Basic frame（rl的基础框架）

**Value based** vs **Policy gradient** 

the policy of DQN is **argmax Q**, which is **discrete**

the policy of PG is **actor network**, which is **continuous**

**Result**：

Policy gradient **lose** at sample efficiency:  DQN is more sample efficient

policy gradient **lose** at stability: gradient of policy could be noisy and high-variance

Policy gradient **win** at smooth update: model update of PG is more smooth

policy gradient **win** at complex and continuous problem: env with non-differentiable reward functions and continuous action space





**Policy gradient** VS **Actor critic**:

the policy update of **PG** is through **bootstrap**: which learn faster 

the policy update of **AC** is through **MC**, which  learn slower

**Result**：

Actor critic **win** at sample efficiency



### Comparasion（rl算法的对比）

| Algorithm | Reward | Convergence Speed | Sample Efficiency | Robustness |
| --------- | ------ | ----------------- | ----------------- | ---------- |
| DQN       | 9      | 8                 | 9                 | 7          |
| A3C       | 8      | 9                 | 6                 | 8          |
| PPO       | 7      | 7                 | 8                 | 6          |
| TRPO      | 7      | 6                 | 7                 | 9          |
| AC        | 6      | 6                 | 6                 | 6          |
| VPG       | 7      | 5                 | 7                 | 7          |
| DDPG      | 8      | 7                 | 9                 | 8          |
| SAC       | 9      | 7                 | 8                 | 9          |

- comparison between ppo and sac

![image-20230504102441608](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504102441608.png)



- comparison between ppo and sac





- DQN

![image-20230504094736482](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504094736482.png)









- A3C

![image-20230504094802502](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504094802502.png)







- ppo

![image-20230504094818441](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504094818441.png)





- trpo

![image-20230504094833674](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504094833674.png)



- AC

![image-20230504094856363](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504094856363.png)



- vpg



![image-20230504095430731](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504095430731.png)

- DDPG

![image-20230504095442568](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504095442568.png)



- sac

![image-20230504095454177](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504095454177.png)













## Questions（问题）

***Q1: why cannot replay buffer be applied to ppo? why max entropy cannot be applied to ppo?***

1. the policy optimization of ppo rely on current policy, ppo use  a rolling buffer to store the most recent trajectories and samples them
2. policy update objective in ppo already includes an entropy term 



***Q2: what is the influence that whether the reward function is differentiable or not in reinforcement learning?***

gradient-based optimization techniques are needed to update the policy and value function based on the observed rewards and states.

***Q3:in neural network ,how much does the update of model influence the prediction of next state, if it matters , maybe we just prefer the most important state?***

***Q4. when update Q,  will it be more stable that the action of next Q is chosen based on the probability distribution of action?\***

![image-20230504112741947](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230504112741947.png)



***Q5 we use epsilon decay or entropy to explore the env, so what if we  mark the action we choose in a table, and next time choose the action we haven't choose?***

***Q6. professor Vinny said MC is not same as sample a  set of trajectories***









## Insights（想法）

- maybe can view sample efficiency, robustness  and convergence from the perspective of machine learning: we want to learn faster and learn from useful information), then maybe absorb some experience from existing papers on gradient descent and machine learning.

  



# Algorithms-multi agent（多智能体算法）

https://jianzhnie.github.io/machine-learning-wiki/#/deep-rl/papers/Overview

centralized critic network

![../_images/MARL_cooperation_algo.png](https://raw.githubusercontent.com/magiclucky1996/picgo/main/MARL_cooperation_algo.png)



Valued-based MARL

roma

Qmix

Actor-Critic MARL

maac

coma

maddpg

mappo





# Upcoming conferences（会议总结）

***later do a complete notion chart: 之后做一个完整的notion表格***

- machine learning

1. Conference on Neural Information Processing Systems (NeurIPS) (A category) https://nips.cc/Conferences/2023/CallForPapers
2. International Conference on Machine Learning (ICML) (A category)  https://icml.cc/
3. Conference on Robot Learning (CoRL)  https://www.corl2023.org/
4. IEEE Intelligent Transportation Systems Conference (ITSC) https://ieee-itss.org/event/itsc2023/
5. Transportation Research Board Annual Meeting (TRB)  https://www.trb.org/AnnualMeeting/AnnualMeeting.aspx
6. International Joint Conference on Artificial Intelligence (IJCAI) (A category)  https://www.ijcai.org/  IJCAI-PRICAI-24: shanghai: out of date for 2023
7. European Conference on Artificial Intelligence (ECAI) https://ecai2023.eu/ECAI2023
8. International Conference on Automated Planning and Scheduling (ICAPS) https://icaps23.icaps-conference.org/ : out of date for 2023
9. International Conference on Learning Representations (ICLR)https://iclr.cc/ out of date for 2023
10. International Conference on Robotics and Automation (ICRA) https://www.icra2023.org/ out of date for 2023
11. International Symposium on Transportation and Traffic Theory (ISTTT) https://limos.engin.umich.edu/isttt25/
12. IEEE Conference on Decision and Control (CDC) https://cdc2023.ieeecss.org/ out of date for 2023
13. IEEE International Conference on Intelligent Transportation Systems (ITSC) https://ieee-itss.org/event/itsc2023/  https://2023.ieee-itsc.org/ out of date for 2023
14. International Conference on Control, Automation and Information Sciences (ICCAIS) http://iccais2023.org/ 
15. International Conference on Control, Automation, Robotics and Vision (ICARCV) https://www.intelligentautomation.network/events-intelligent-automation/agenda-mc?utm_campaign=27031.007_BLUE_GPPC&extTreatId=7576989&gclid=Cj0KCQjw6cKiBhD5ARIsAKXUdyY_SSzgGhuf4T7L6NxsscqfgI6HypsBEBtUoK1KGE28nwelmmOX-oIaAvveEALw_wcB



- traffic 

1. World Conference on Transport Research Society (WCTRS) http://wctr2023.ca/
2. International Association of Traffic and Safety Sciences (IATSS)
3. IEEE Intelligent Vehicles Symposium (IV) https://2023.ieee-iv.org/
4. Transportation Science and Logistics Society (TSL)
5. International Conference on Transport and Health (ICTH)
6. International Symposium on Transportation Network Reliability (INSTR) https://easychair.org/cfp/instr2023
7. IEEE International Conference on Intelligent Transportation Systems (ITSC)
8. European Transport Conference (ETC) https://aetransport.org/etc
9. International Conference on Traffic and Transport Psychology (ICTTP) ICTTP 8 2024 Tel Aviv, Israel.
10. ITS World Congress https://itsworldcongress.com/ 2024 dubai



- smart city 

1. IEEE International Smart Cities Conference (ISC2)
2. ACM International Conference on Ubiquitous Computing and Communications (UbiComp)
3. International Conference on Smart Cities and Green ICT Systems (SMARTGREENS)
4. Smart Cities Symposium Prague (SCSP)
5. IEEE International Conference on Smart City Innovations (SCI)
6. International Workshop on Smart Cities and Urban Analytics (UrbanGIS)
7. International Conference on Smart Data and Smart Cities (SDSC)
8. Smart City Symposium (SCS)
9. International Conference on Sustainable Smart Cities and Territories (SSCt)
10. European Conference on Smart Objects, Systems and Technologies (Smart SysTech)



- intelligent connected vehicle

1. International Conference on Connected Vehicles and Expo (ICCVE)
2. IEEE Vehicular Technology Conference (VTC)
3. International Conference on Vehicle Technology and Intelligent Transport Systems (VEHITS)
4. IEEE Conference on Control Technology and Applications (CCTA)
5. International Conference on Vehicle Engineering and Intelligent Transportation Systems (VEITS)
6. IEEE International Conference on Connected and Autonomous Vehicles (ICCAV)





#### May

**May 8**

ECAI 2023 https://ecai2023.eu/ECAI2023

**May 11**

NIPS 2023  https://nips.cc/Conferences/2023/CallForPapers

**May 15**

ITSC 2023 https://2023.ieee-itsc.org/





#### June

**June  8** 

CoRL 2023 https://www.corl2023.org/



#### July

**July 15**

ICCAIS 2023 http://iccais2023.org/



#### August

**August 1**

TRB 2024 https://trb.secure-platform.com/a/page/TRBPaperReview#Instructions



#### Sepetmber



#### October



#### November



#### December







## note



1. #### Trpo PPO MAPPO

1.1 Trpo

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







### 1.2 PPO

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



2. ### pieter abbeel rl course

https://www.youtube.com/watch?v=2GwBez0D20A&t=130s

2.1 : MDP

***insight: group of robot learn faster: data sharing, more efficient sampling of the env***



- groups of robots learn faster, they can share date, more efficient sampling of the env, save wall time
- gamma (discount factor) is also designed based on what our goal is, if we want the agent of care more about things happen in closer steps, then ...
- if gamma is introduced, state take less  steps to the reward is with higher value, it's like the "time" of game world. but it should not be same as the future evaluated in our real world.
- update: in grid world,  we swap a time for all the grid , what if we use different way to swap all the states?
  - like along trajectory
  - like importance sampling
- Discount factor influence convergence: 0: faster 1: longer
- why it converge

![image-20230426134554214](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426134554214.png)





##### effect of discount and noise







![image-20230426150334001](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426150334001.png)

(a)

![image-20230426171844052](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171844052.png)



(b)

![image-20230426171913044](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171913044.png)



(c)

![image-20230426171946200](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426171946200.png)





(d)

![image-20230426172016159](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426172016159.png)







- update for Q*  , as default,  the agent thereafter acting optimally

![image-20230426172142184](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426172142184.png)





- policy evaluation



### Max entropy





- how do we collect data
  - use current policy to collect data	, if policy is deterministic , data collection would not be interesting
  - with entropy, policy will be with more variation in how the data is collected  





- entropy 



![image-20230426173232834](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230426173232834.png)









- a distribution over near-optimal solution
  - **robust policy**:  the env could change, if it's distribution instead of deterministic it's more robust
  - **robust learning** : we can keep collecting exploratory data during learning 

- collect along learning, or collect with exploration , then off-policy update

- insights: how about when the best action changed in a state , increase the   possibility of explore and update to former states,( like in my last paper, in board are , increase explore, in narrow area, reduce explore)
- insights:  after update the value of a state , trace back and update all the former state (read the trace chapter of sutton book)



### 2.2 : Q learning

**properties**

- converge even if act suboptimal (epsilon greedy)
- epsilon decay: if not do decay , latest experience will make you hop around
- epsilon: u need to make it small eventually 
- epsilon: cannot decay too fast: cannot update enough

**requirement**

- state and actions are visited infinitely often: doesn't matter how u select actions
- learning rate schedule: 
  - reference: [On the Convergence of Stochastic Iterative Dynamic ...](https://dspace.mit.edu/bitstream/handle/1721.1/7205/AIM-1441.pdf?sequence=2)

![image-20230501182944684](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230501182944684.png)





### play with rl



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



### *paper reading list:* 

- *playing atari with drl* https://arxiv.org/abs/1312.5602
- *rainbow*  https://arxiv.org/abs/1710.02298
- *ppo* https://arxiv.org/abs/1707.06347
- *mappo* https://arxiv.org/abs/2103.01955
- *maven* https://arxiv.org/abs/1910.07483
- qmix https://arxiv.org/abs/1803.11485
- maddpg https://arxiv.org/abs/1706.02275
- coma https://arxiv.org/abs/1705.08926










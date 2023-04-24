# play with gym env

## env1: cart pole

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





## env2 lunar

安装

pip install box2d-py





## env3 taxi

https://huggingface.co/learn/deep-rl-course/unit3/introduction?fw=pt
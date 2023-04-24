# Part1: Codes implementation(MARL)

## 1. Ubuntu setting

### Install oh my zsh

```bash
sudo apt-get install zsh
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
source ~/.zshrc

```

### setting of input method of ubuntu

https://www.zhihu.com/question/418042828



### Install nvidia driver

driver version : `470.63.01`.

CUDA version: CUDA Toolkit 11.4.



If your project is written with TensorFlow 1 and you need to use it with the newer versions of CUDA and NVIDIA driver that you have installed, you have a few options:

1. Downgrade your NVIDIA driver and CUDA Toolkit versions to be compatible with TensorFlow 1. You can follow the installation instructions for TensorFlow 1.15.4 with GPU support, which requires CUDA Toolkit 10.0 or 10.1 and NVIDIA driver version 418.x or higher.
2. Use a virtual environment or container with the specific versions of CUDA Toolkit, NVIDIA driver, and TensorFlow 1 that your project requires. You can create a new virtual environment or container with the appropriate dependencies using tools like virtualenv or Docker.
3. Upgrade your project to use TensorFlow 2.x. Although there are some differences between TensorFlow 1 and TensorFlow 2, many of the core concepts and functionalities are similar, so the migration process may not be too difficult. You can use TensorFlow's migration guide to help you with the process.

​	

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-<VERSION>
```



### meet problem with unfigured nvidia driver

```bash

sudo apt-get remove --purge nvidia-dkms-470 nvidia-driver-470
```



### still doesn't work

```bash

dpkg -l | grep -i nvidia
sudo apt-get remove --purge '^nvidia-.*'

```



### Problem： gpu driver -> install nvidia driver470

```bash
dpkg: error processing package nvidia-driver-470 (--configure):
 dependency problems - leaving unconfigured
No apport report written because the error message indicates its a followup error from a previous failure.
                                                                                            Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...
Processing triggers for gnome-menus (3.36.0-1ubuntu3) ...
Processing triggers for libc-bin (2.35-0ubuntu3) ...
Processing triggers for man-db (2.10.2-1) ...
Processing triggers for mailcap (3.70+nmu1ubuntu1) ...
Processing triggers for initramfs-tools (0.140ubuntu13.1) ...
update-initramfs: Generating /boot/initrd.img-5.19.0-38-generic
Errors were encountered while processing:
 nvidia-dkms-470
 nvidia-driver-470
E: Sub-process /usr/bin/dpkg returned an error code (1)
```

- solution: set in the software and update of ubuntu and reboot





### Uninstall old Cuda

```bash
cd /usr/local/cuda/bin
./cuda-uninstaller
```





### Install Cuda11.4

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.3/local_installers/cuda_11.4.3_470.82.01_linux.run
## test the install of cuda
nvcc --version
```



##### Install cudnn

- https://developer.nvidia.com/cudnn

test the install of cudnn

***









## 2. MARL env setting

#### 1. install StarCraft II env

##### Reference

- https://github.com/magiclucky1996/on-policy

  - ```unzip SC2.4.10.zip```
  - `echo "export SC2PATH=~/StarCraftII/" > ~/.bashrc`
  - download SMAC Maps, and move it to `~/StarCraftII/Maps/`.
  - To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

- https://github.com/oxwhirl/smac

  - Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

  

- https://github.com/oxwhirl/pymarl/blob/master/install_sc2.sh

#####  Detailed steps

- follow the instruction of shell script below

``````
# download and install starcraft ii

wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip

# download smac map

MAP_DIR="$SC2PATH/Maps/"
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR

``````



##### Problem1  wandb.init

	run = wandb.init(config=all_args,
	                 project=all_args.env_name,
	                 entity=all_args.user_name,
	                 notes=socket.gethostname(),
	                 name=str(all_args.algorithm_name) + "_" +
	                      str(all_args.experiment_name) +
	                      "_seed" + str(all_args.seed),
	                 group=all_args.map_name,
	                 dir=str(run_dir),
	                 job_type="training",
	                 reinit=True)

- these attributes

  - project

  - env_name

  - entity

  - user_name

  - notes

  - name

  - gourp

  - dir

##### Ideas 

1. wandb permission denied: modify the wandb.init 

2. python 10 is not compatible: switch to python 3.6 (comply with origin env as much as possible)
3. finsh the implementation of four marl project and get some figures.
   1. understand detail of each project
4. finish my explorement of fundamental rl algorithms



***

















# Part 2: Reading

## Reference

- **spinning up**

https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html (there some recommendation of  reading here)

https://spinningup.openai.com/en/latest/spinningup/keypapers.html?highlight=rainbow (key papers of rl)

https://spinningup.openai.com/en/latest/spinningup/keypapers.html?highlight=rainbow#model-free-rl

- **github paper collection**

https://github.com/LantaoYu/MARL-Papers





## Papers



### 1. Playing Atari with Deep Reinforcement Learning

为什么需要样本是独立的，独立同分布到底是什么意思





### 2. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks

https://arxiv.org/abs/2006.07869

*June 2020  ; Georgios Papoudakis ； Phd student, ；School of Informatics ； university of Edinburgh,*

- experiment results serving as reference
- insights regarding the effectiveness of different learning approaches
- open-source algorithm codebase
- open-source two rl envs

#####  reading

- algotithms

  - independent
    - iql
    - ia2c
    - ippo
  - ctde
    - ma ddpg
    - coma
    - ma ac
    - ma ppo
  - value decompose
    - vdn
    - qmix

- envs

  - Repeated Matrix Games
  - Multi-Agent Particle Environment(MPE)
  - StarCraft Multi-Agent Challenge
  - Level-Based Foraging(LBF)
  - Multi-Robot Warehouse

  



##### some thinking in the reading

- is there insights between different games? 不同类型的game有不同的特点

- 不同问题对joint action的需求不一样，这个要对问题的机理进行分析

- 运行时间：有的时候时间不一定花在算法的时间上，可能花在和环境交互的时间上，不同大小的模型，更新频率步长等也会影响算法时间，需要弄清楚算法时间相关影响变量。

  - to be explored
  - if we used fixed algorithm, 随机性从哪里来，我能否限制随机性来对环境做比较。还是说我跑很多次，取平均值，来做比较。首先你得清楚整个训练的每个步骤和影响因素，所以要细读代码。（之后多攒一些问题可以专门地找人咨询）

- it could be interesting to try the chase and hide env

- 做成库给人用确实是很好的工具，我想做我自己的库，但是先看看有没有已有的（我先找点库跑一跑）

- 拿到其他智能体的模型可能很难直接从模型参数得到啥，但是可以输入看 模型输出的结果，（模型的结果，到环境的演化的概率分布）

  



##### conclusion(what i learn from this passage):

- 



### 









### 3. Rainbow: Combining Improvements in Deep Reinforcement Learning

https://arxiv.org/abs/1710.02298

*Oct 2017*

*Matteo Hessel*

*Deepmind*



### 4. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games

https://arxiv.org/abs/2103.01955

*Nov 2022*

*chao yu*



















## Books

### 1. RLbook by sutton\\\









## Videos

### 1. Multi-Agent Reinforcement Learning (Part I)

https://www.youtube.com/watch?v=RCu-nU4_TQM

> simon institute; chi jin, princeton



### 2. Shusen wang













# Part 3: Questions

## 1. why data needs to be IId?

> 可以参考分布式机器学习的一些东西，（我一直在做分布式，分布式机器学习，分布式存储）

https://www.youtube.com/watch?v=STxtRucv_zo&t=1504s

1/ 具有相同的概率分布：分布没有波动

2/ 相互独立：了解一个变量的值不会提供另外一个变量的信息



##### Example 1[[edit](https://en.wikipedia.org/w/index.php?title=Independent_and_identically_distributed_random_variables&action=edit&section=8)]

A sequence of outcomes of spins of a fair or unfair [roulette](https://en.wikipedia.org/wiki/Roulette) wheel is i.i.d. One implication of this is that if the roulette ball lands on "red", for example, 20 times in a row, the next spin is no more or less likely to be "black" than on any other spin (see the [Gambler's fallacy](https://en.wikipedia.org/wiki/Gambler's_fallacy)).

A sequence of fair or loaded dice rolls is i.i.d.

A sequence of fair or unfair coin flips is i.i.d.

In [signal processing](https://en.wikipedia.org/wiki/Signal_processing) and [image processing](https://en.wikipedia.org/wiki/Image_processing) the notion of transformation to i.i.d. implies two specifications, the "i.d."part and the "i." part:

(i.d.) the signal level must be balanced on the time axis;

(i.) the signal spectrum must be flattened, i.e. transformed by filtering (such as [deconvolution](https://en.wikipedia.org/wiki/Deconvolution)) to a [white noise](https://en.wikipedia.org/wiki/White_noise) signal (i.e. a signal where all frequencies are equally present).

##### Example 2[[edit](https://en.wikipedia.org/w/index.php?title=Independent_and_identically_distributed_random_variables&action=edit&section=9)]

Toss a coin 10 times and record how many times does the coin lands on head.

1. Independent – each outcome of landing will not affect the other outcome, which means the 10 results are independent from each other.
2. Identically Distributed – if the coin is a homogeneous material, each time the probability for head is 0.5, which means the probability is identical for each time.

##### Example 3[[edit](https://en.wikipedia.org/w/index.php?title=Independent_and_identically_distributed_random_variables&action=edit&section=10)]

Roll a dice 10 times and record how many time the result is 1.

1. Independent – each outcome of the dice will not affect the next one, which means the 10 results are independent from each other.
2. Identically Distributed – if the dice is a homogeneous material, each time the probability for the number 1 is 1/6, which means the probability is identical for each time.

##### Example 4[[edit](https://en.wikipedia.org/w/index.php?title=Independent_and_identically_distributed_random_variables&action=edit&section=11)]

Choose a card from a standard deck of cards containing 52 cards, then place the card back in the deck. Repeat it for 52 times. Record the number of King appears

1. Independent – each outcome of the card will not affect the next one, which means the 52 results are independent from each other.
2. Identically Distributed – after drawing one card from it, each time the probability for King is 4/52, which means the probability is identical for each time.





## 2. why machine learning need IID?

Machine learning uses currently acquired massive quantities of data to deliver faster, more accurate results.[[7\]](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables#cite_note-7) Therefore, we need to use historical data with overall representativeness. If the data obtained is not representative of the overall situation, then the rules will be summarized badly or wrongly.

Through i.i.d. hypothesis, the number of individual cases in the training sample can be greatly reduced.

This assumption makes maximization very easy to calculate mathematically. Observing the assumption of independent and identical distribution in mathematics simplifies the calculation of the likelihood function in optimization problems. Because of the assumption of independence, the likelihood function can be written like this



In order to maximize the probability of the observed event, take the log function and maximize the parameter *θ*. 

The computer is very efficient to calculate multiple additions, but it is not efficient to calculate the multiplication. This simplification is the core reason for the increase in computational efficiency. And this Log transformation is also in the process of maximizing, turning many exponential functions into linear functions.

For two reasons, this hypothesis is easy to use the central limit theorem in practical applications.

1. Even if the sample comes from a more complex non-Gaussian distribution, it can also approximate well. Because it can be simplified from the central limit theorem to Gaussian distribution. For a large number of observable samples, "the sum of many random variables will have an approximately normal distribution".
2. The second reason is that the accuracy of the model depends on the simplicity and representative power of the model unit, as well as the data quality. Because the simplicity of the unit makes it easy to interpret and scale, and the representative power + scale out of the unit improves the model accuracy. Like in a deep neural network, each neuron is very simple but has strong representative power, layer by layer to represent more complex features to improve model accuracy.



some user has photo full of animals, some user has photo full of views

iid：

Independent and identically distributed: the data is uniform, randomly disrupted, and the statistics of each node are similar (mean, variance)

If the data is disrupted, shuffle, the data becomes independent and identically distributed, which is equivalent to a node, a distribution within a set

The statistical nature of each mobile phone user's data is different. Some people like to take pictures of landscapes, while others like to take selfies.

My understanding: to prevent crooked data science? Go up the hill evenly, don't click left and right?



## 3. why Experience replay

> https://www.youtube.com/watch?v=rhslMPmj7SY&list=RDCMUC9qKcEgXHPFP2-ywYoA-E0Q&index=6













## 4. the reuse of experience in experience replay

first thing: how is transition used?





each experience is a sampling of the real world,

if we want to reuse it , why not 







## 5. what is sampling efficiency in rl?

what is meaning of sampling efficiency：

sampling efficiency：

unbalance load, the amount of some user's data is large, others are small so that some node has iterated hundreds of time, some just one time











## 5. Q-learning take Td-error  as loss func, how about AC and PG?

- **this is the leaning of td**

![image-20230418164141297](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/image-20230418164141297.png)









- **Policy Gradient**

Policy gradient：Instead of minimizing TD error, it maximizes V, maximizes V, directly calculates the derivative of the model parameters on A, and then performs gradient ascent to update the model parameters to maximize V,



- the update of policy in *Policy Gradient**

![image-20230418163835203](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/image-20230418163835203.png)



- while the v is equal to :

![image-20230420102406349](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420102406349.png)

- then calculate the derivative

![image-20230420102508795](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420102508795.png)





- use chain rule to calculate form 2

![image-20230420102536383](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420102536383.png)



- so the result is :



![image-20230418163957262](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/image-20230418163957262.png)

- for form 1 : at each state, summation over all actions, could also be replaced with MC 
- for form 2: Hard to calculate expectation, use MC instead

- 











- **overview of the algorithm**

![image-20230418174000255](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/image-20230418174000255.png)

**questions**

- Question1: if use form 1: how to scan over all actions?

- Question2: if use form 2: how to guarantee adequate sampling?





## 6. how about sampling multiple actions and update the model at the same state?(like we do enough sampling at the same state in multi-armed bandit 



because of the independence of data?

we know that related data and data with same distribution is harmful  











## 7. how to  Design traffic control problem

fully-observed:

1. the action of other agents 
   1. how to utilize the action of other agent
2.  the strategy of other agents 
   1. how to utilized the strategy of other agents: prediction
3. the state of other agents



the design of global reward function



the design of action in different levels of traffic





the source of random: random form the strategy; random from the state transferring of the env





***



# Part4: projects



# Project1: Mappo official implementation

https://github.com/magiclucky1996/on-policy

#### paper reading

https://arxiv.org/abs/2103.01955



### 







#### bugs when running

- RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`



### running

- first running

































***







# Project2: light mappo

https://github.com/magiclucky1996/light_mappo







***







#  Project 3: Marl-sumo

https://github.com/magiclucky1996/deeprl_signal_control



### install sumo

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

```





#### Problem1: with sumo cannot find local schema

- warning

```bash
Warning: Cannot read local schema '/usr/share/sumo/bin/data/xsd/additional_file.xsd', will try website lookup.
Warning: Cannot read local schema '/usr/share/sumo/bin/data/xsd/routes_file.xsd', will try website lookup.
Warning: Cannot read local schema '/usr/share/sumo/bin/data/xsd/net_file.xsd', will try website lookup.
```

- solution

change sumo version:turn to sumo 1.16, 

#### Problem2: not using gpu

- solution

reinstall all the staff related to gpu

after install driver, cuda, if they are set the right env variable?: this may cause not working, so now im running with cpu, i show reinstall and do a test.

#### Problem 3:  Code is written in tensorflow 1 but gpu only support tensorflow  2 

1. modify the code 

Replace all `import tensorflow as tf` statements with `import tensorflow.compat.v1 as tf` followed by `tf.disable_v2_behavior()`.

Replace all deprecated TensorFlow 1.x syntax with TensorFlow 2.x syntax. This includes changes to `tf.Session()` to `tf.compat.v1.Session()`, `tf.global_variables_initializer()` to `tf.compat.v1.global_variables_initializer()`, and so on.

Replace `tf.contrib` modules with equivalent modules in `tf.keras` or other TensorFlow 2.x modules. For example, `tf.contrib.layers` can be replaced with `tf.keras.layers`.



2. migrate following the official instruction

https://www.tensorflow.org/guide/migrate



#### Problem4: set the growth of gpu

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
```









## MARL-SUMO Modeling

### Problem definition







### Markov decision process modeling









### System informations



``` as sdf
system: ubuntu 22.04 LTS
python: 3.8
gpu: gtx 3060 laptop version
cuda: 11.3
cudnn: version correspongding to cuda version
SUMO: 1.16
tensorflow: tensorflow-gpu 1.14
```



###  bugs and problems





## Training record

##### Q-leaning with 16 intersections

1. 0413 morning

- hyperparameter

```max_grad_norm = 40
gamma = 0.99
lr_init = 1e-4
lr_decay = constant
epsilon_init = 1.0
epsilon_min = 0.01
epsilon_decay = linear
epsilon_ratio = 0.5
num_fc = 128
num_h = 64
batch_size = 20
buffer_size = 1000
reward_norm = 3000.0
reward_clip = 2.0
```

- experiment time:  13h

- experiment result:

  

##### ![iqld](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/iqld.png)

1. 0413 morning

- hyperparater

  ```
  [MODEL_CONFIG]
  rmsp_alpha = 0.99
  rmsp_epsilon = 1e-5
  max_grad_norm = 40
  gamma = 0.99
  lr_init = 5e-4
  lr_decay = constant
  entropy_coef_init = 0.01
  entropy_coef_min = 0.01
  entropy_decay = constant
  entropy_ratio = 0.5
  value_coef = 0.5
  num_fw = 128
  num_ft = 32
  num_lstm = 64
  num_fp = 64
  batch_size = 120
  reward_norm = 2000.0
  reward_clip = 2.0
  ```

- result

  ![ac413](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/ac413.png)

- sac run before

  

  ![ac](https://raw.githubusercontent.com/magiclucky1996/picgo/main/test/ac.png)

  

  

  

  

# Project 4: General marl

https://github.com/magiclucky1996/MARL-code-pytorch







***



# Part 5: what's next?

## 1. what's next? Incorporate Mappo into sumo simulation



- actually this is similar to what we do two years ago, combine sac with sumo simulation
- but this time, dive deeper into problem definition, modeling, network , rl learning algorithm part.



## 2. what's next? go through rl basics





## 3. what's next? go through markov modeling





##  4. what's next? go through urban traffic modeling and control



每次只能解决一个问题，按重要度进行排序，我已经看够了各种理论，今天重要的是找几个算法和几个环境玩一玩，玩玩不同的环境，找出环境之间的feature，得到一些insights





然后把ppo理解，看懂mappo，跑起来mappo



明天重点就是交通环境的建模，明天晚上要弄出周四meeting的东西

rl-training： general view



1. 环境初始化，得到一个初始状态（每次都一样对不对，这个初始出发的学习路线可能会不一样，）
2. 我从初始状态出发，开始我的旅行，我在单个状态是通过trail and error进行学习，我在打游戏的时候可以从一个节点重来，但可惜的是现实生活不行，但如果我们把它做成游戏一样不是很好嘛，比如在交通信号灯问题里，，一个状态的试错学习，多试试遥杆，其实控制也是，因为我们没法计算，但是我们可以计算和感觉相结合。连续单状态地更新模型有什么好处？不知道，

## 5. play with envs(different envs and different libs)





















***



# Part6 : Additional information

## 1. Deploy blog with hexo, typora,github and picgo

- install nodejs
- install hexo 

```bash
npm install -g hexo-cli
```

- create a blog

```bash
hexo init my-blog
```

- change to this blog

```bash
cd my-blog	
```

- install dependence

```bash
npm install
```

- install hexo deployer

```bash
npm install hexo-deployer --save
```

- modify the _config.yml

```yaml
url: https://<your github user name>.github.io/



deploy:
  type: git
  repository: git@github.com:username/username.github.io.git
  branch: master

```

- link local git with github so that you can push local file to github

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"

cat ~/.ssh/id_ed25519.pub
```

- copy this pub key to your github account: settings -> ssh and gpg keys
- write your blogs in format of .md and add it to .../hexo/my-blog/source/_posts
- use hexo generate to generate html files

```bash
hexo generate
```

- use hexo server to view locally 

```bash
hexo server
```

- use hexo deploy to deploy to github

```bash
hexo deploy
```





## 2. Version control for blog source with multiple devices by git and github

- on one computer

```bash
cd ~/hexo/my-blog/source/_posts
git init
git add .
git commit -m "first commit on linux"
git remote add origin <link of github repo>
git push -u origin main
```

- on another computer

```bash
cd ~/hexo/my-blog/source
git clone <link of github repo>
git pull origin main # pull from remote repo
git push -u origin main # pull to the remote repo
```



##  3. Image uploader: Picgo and typora

##### reference 

- https://support.typora.io/Upload-Image/

- https://picgo.github.io/PicGo-Core-Doc/zh/guide/config.html#%E9%BB%98%E8%AE%A4%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6

#####  Detailed steps

- in typora
  - click file -> preference -> image: choose download to install picgo-core
- ``picgo set uploader``
- ``picgo use uploader``
- you can set automatically upload photo in typora -> preference -> image
- it's done! enjoy!







tips:

we don't need to run it by ourselves if the graph run by others is enough for our need.

last hour for the modeling of marl env

if have a goal or a question to figure, it will motivate , so seperate it into some detailed small questions, and motivate urself, so there will be postive feedback from exploring.

basic ideas of rl -> ppo -> mappo , ...

stably read some papers and incorprate into it





# MARL modeling

## traffic control + ma2c

- problem definition		

![image-20230420114007282](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420114007282.png)



- state space

![image-20230420111112341](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420111112341.png)

- action space

  - phase switch
    - 

  - phase duration
    - make decision for how long the phase last

  - phase itself
    - fixed control period



- reward function 

![image-20230420111340901](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230420111340901.png)



- training algorithm



## SMAC + mappo











# Rl algorithm

## trpo

- general view

In order to solve the problem of  leaning rate being too big, the cost function is approximated in trust region, and then the optimal value is calculated 







- persudo code









- code









## ppo

- general view
- persudo code
- code







## mappo

- general view
- persudo code
- code


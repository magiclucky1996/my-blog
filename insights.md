# insights

**td error 先不更新，把式子放在那里，等到达最后再更新。把已探究出来的列着，等到回合结束再更新。**



**argmax为什么不好？为什么actor比argmax更好？为什么ac就比q learning好？**





**为什么不能一开始就疯狂探索，100%，反正我可以不用管当前的性能？**



- 我可以一开始就疯狂探索，然后把状态转移的r放在那里放着，
- gym的rl环境不够准确，我觉得不同的环境千差万别。。没法体现rl算法的效率。这些人也许并没有多高明。

This approach is called pure exploration and exploitation (PEE) and can be used in some cases where exploration is very costly or where the environment is very simple. However, in most real-world scenarios, PEE is not an optimal approach.

The problem with PEE is that the agent spends a lot of time exploring and collecting data, but not enough time exploiting that data to improve its policy. This can lead to slow learning and poor performance, especially in complex environments where there are many possible actions and states.

In contrast, most RL algorithms use a balance between exploration and exploitation, where the agent takes actions that are likely to yield high rewards based on its current policy while also occasionally exploring new actions or states. This allows the agent to learn quickly while still exploring new possibilities, leading to faster learning and better performance.

Furthermore, in many RL problems, the environment is dynamic and can change over time. In such cases, it is important for the agent to continuously explore and adapt to changes in the environment to maintain optimal performance. This requires a balance between exploration and exploitation, as well as the ability to update the policy based on new data and experiences.

In summary, while PEE can be a useful approach in some cases, a balanced approach between exploration and exploitation is generally more effective for most RL problems.



**为社么discount factor**

unstable env, noise in reward, 

**为什么神经网络代替q表，那部分要用神经网络学习的不能表征的“规律”是什么？**

如果q表很大，就很难搞，为啥，那就建一个大的q表呗





**策略和价值的本质区别是什么，是不是一样的，那个策略更新跟argmax到底是不是一样的**





后面我可以i自己和环境玩，但是先按照教程完整拼一遍，






### what to do today:

make decisions for what to do and do it .

1. make the pre for next week

2. finish the student work (no i wish)

3. finish reading for rl and marl(god)
4. buy shoes and belt and maybe towel

### specific direction

direction 1 : hierarchical traffic control

direction 2 : online learning



### survey for traffic control

- ask chatgpt to give the survey on traffic control

  - way of control

    - traditional: off-line: get a offline strategy, and implement it 
    - dynamic: adjust based on real-time data

  - systems in industry

    - australia: SCATS
    - england: SCOOT
    - america: RHODES
    - italy: SPOT/UTOPIA
    - China: NUTCS SMOOTH

  - systems currently used 

    - ireland: SCATS

    - singapore: UTC (SCOOT algorithm)

    - Japan: TCIS (ALINEA（Adaptive Line Enhancement Algorithm）algorithm)

    - England: scoot MOVA UTC

    - Germany: centralized control : Zentrale Verkehrsleittechnik”（ZVL）

      - scale: coordinate the whole country and cities

      - method:

      - resource

        - Websites of traffic management companies:

        - Siemens Mobility: h
        - ttps://www.siemens-mobility.com/de/de/themen/verkehrstechnik/verkehrsmanagement/
        - Swarco Traffic Systems: https://www.swarco.com/de/Verkehrssysteme/Verkehrsmanagement
        - Imtech Traffic & Infra: https://www.imtechtraffic.com/de/

        1. Government websites:

        - Federal Ministry of Transport and Digital Infrastructure (BMVI): https://www.bmvi.de/SharedDocs/DE/Home.html
        - State transportation agencies (examples):
          - Hessen Mobil: https://www.mobil.hessen.de/
          - Landesbetrieb Straßenbau NRW: https://www.strassen.nrw.de/

        1. Research databases:

        - Google Scholar: https://scholar.google.com/
        - IEEE Xplore: https://ieeexplore.ieee.org/Xplore/home.jsp

        1. Professional associations:

        - German Association for Traffic and Transportation (DVWG): https://www.dvwg.de/
        - German Road and Transportation Research Association (FGSV): https://www.fgsv.de/

    - US: scats, scoot, ATMS (Advanced Traffic Management System) and TSP (Transit Signal Priority).

    - China: NUTCS SMOOTH iurban I-sense

  - frames and algorithms in currently used system

    - **SCOOT**	

      - **"reserach on the transyt and scoot methods of signal coordination": key ideas of transyt and scoot**

        - Dennis I. Robertson, 1986

        - https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=dfefb6e8b96b19ecb7693619fc92530e0a95e7a2

        - genius of rrl invented combination method of coordinating signals

          - reference: glasgow experiment in area traffic control
          - traffic model estimated average queue at all stop line 
          - sum of queues are the optimization goal, 

        - for optimization , **goal** should be clear, and **queue** as goal is good 

          - sum of queues
          - if u optimize sum of queue, delay also tend to be lower
          - optimization goal also take stop times
            - waste energy; cause danger
            - weight factor balance **queue** and **stops**
              - when queue is optimized, stops are also optimized, when stops are taken valued, agent tend to prolong cycle times

        - **many engineers maximize bandwidth in time-distance graph, but bandwidth is not good when**

          - not possible to calculate financial term
          - when congestion , bandwidth  concept starts to break down
            because the growth of queues disrupts
            the bands
            - bandwidth:  maximum flow rate that can be achieved while maintaining a reasonable speed and avoiding congestion.
          - signal coordination on a fixed time plan
            - ![image-20230514185002902](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230514185002902.png)

        - Transyt

          - have real-time problem to use real-time data

        - **scoot principle**

          - measure CFP
          - online traffic model
          - ![image-20230514191147690](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230514191147690.png)
          - incremental optimization
            - elastic coordination
              - before phase change, three options:
                - 1. advance  change for 4 seconds
                  2. retard change for 4 seconds
                  3. unchanged

          

    - **optimizing networks of traffic signals in real time- the scoot method**: how scoot evloves

      - Dennis I. Robertson, 1991
      - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=69966&tag=1
      - firstly : utc system,
        - time distance graph
        - ![image-20230514162227682](https://raw.githubusercontent.com/magiclucky1996/picgo/main/image-20230514162227682.png)
        - ideal green wave: in the time,distance diagram, it is hard to estimate the queues with TD, then there starts to be **transyt**
        - **transyt**:
          - developed by TRRL
          - prediction: assume traffic platoons travel at a known speed with some dispersion(一定分布下定速行驶), queue discharge full rate during green time(一定速率驶出路口)
          - one year to get a plan for offline solution
        - princeple of scoot
          - mearsure cfp in real time
          -  update online model of queue
          - incremental optimization of signals
        - information of traffic control mode in toronto
          - https://www.toronto.ca/services-payments/streets-parking-transportation/traffic-management/traffic-signals-street-signs/traffic-signals-in-toronto/mode-of-control/
        - TRL software
          - https://trlsoftware.com/products/traffic-control/scoot/
          - traffic simulation software : VISSIM , S-Paramics, sumo, aimsun, transmodeler
          - road sumulation: **ARCADY、PICADY 和 OSCADY**
          - 

    - ALINEA

      - predictive and control

    - **SCATS**

    - RHODES

    - SPOT/UTOPIA

  - conclusion for algorithms used 

    - optimization method: OPAC, prodyn: ()
    - fuzzy control(approximate the strategy)
    - revolution method (search in strategy space)
    - reinforcement learning

  - imagine the easiest way: simple traffic flow , like the water,

  

- ask chatgpt to give the survey on traffic control with rl

- ask chatgpt to recommend surveys on traffic control

- ask chatgpt to recommend surveys on traffic control with rl

### the structure of my pre, first gather information, later make the graph

1. what problems there are : 
   1. given urban traffic, control the traffic by intersection control
   2. 
2. what has been done for each problem
3. in which way they fix each problem:
4. what has not been done
5. what we could do:

### rl system for urban traffic

- preask chatgpt
  - "Multi-agent reinforcement learning for traffic signal control" by Wiering et al. This paper proposes a multi-agent reinforcement learning approach for traffic signal control, which has been shown to outperform traditional signal control methods in large-scale simulations.
    - https://ieeexplore.ieee.org/document/6958095
    - survey: reference 5
    - 
  - "Deep reinforcement learning for traffic signal control" by El-Tantawy et al. This paper proposes a deep reinforcement learning approach for traffic signal control that uses a convolutional neural network to represent the state of the traffic network.
  - "A survey of reinforcement learning applications in traffic signal control" by Li et al. This paper provides a comprehensive survey of the various reinforcement learning techniques that have been applied to traffic signal control, along with their strengths and limitations.
  - "Distributed reinforcement learning for adaptive traffic signal control" by Ma et al. This paper proposes a distributed reinforcement learning approach for traffic signal control that allows for coordination between traffic signals at different intersections.
  - "Machine learning for traffic signal control: A review" by Ma et al. This paper provides a review of the various machine learning techniques that have been applied to traffic signal control, including supervised learning, unsupervised learning, and reinforcement learning.

- as far as  i can remember
  - thousand of junctions
  - tianshu chu
  - my master thesis: some classical papers



**what is the easiest relationship between the traffic phase of intersctions**

- give a influence on future queue of next intersection , and because there are two options , so the decision-making is either benefical or harmful, difference would be how much it is , or it is actually zero.
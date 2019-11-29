# Hierarchical Learning and Planning (in progress)

---

## Project Overview

### Difficulties in hierarchical decision making: 

* Efficiently reusing known sub-policies: __meta-learning__
* Enabling __smooth (dis-)engagements__ between high-level policies. In unknown environments or surroundings, hierarchical approaches tend to oscillate between the high- and low-level strategy 
and are thereby stuck in undesirable situations
* Finding suitable __representations__ for different abstraction levels. Should the perception-planner interface be handcrafted at all?
 
### Agents

#### Clipped Deep Q-Learning (DQL)
Primarily used as a non-hierarchical baseline implementation, but can can futher be used as part of the heirarchical planning framework.

#### Hierarchical DQL
(ToDo)

#### Successor Reinforcement Learning
* [Deep Successor Feature Learning](https://arxiv.org/pdf/1606.02396.pdf): "Deep Successor Reinforcement Learning", T. Kulkarni, et al., 2016.
* [Successor Feature Learning](https://arxiv.org/pdf/1906.09480.pdf) in POMDPs: "A neurally plausible model learns successorrepresentations in partially observable environments", E. Vértes, M. Sahani, 2019.

### Hierarchical SR Learning
(ToDo)

---

### How to run the experiments:

Requirements:
* Python 3.6+
* PyTorch 1.3+
* [Mini Gridworld](https://github.com/maximecb/gym-minigrid), a minimalistic gridworld environment developed by [maximecb](https://pointersgonewild.com/about/)

Installation:
* Create a virtual python environment and install  all requirements with: `bash install.sh`
(The requirements can also be installed separately with `pip3 install -r requirements.txt`)

Experiments for a particular environment can be run using:

```
python train.py
-e    --environment   to choose one of the pybulletgym environments. Default is "InvertedDoublePendulumMuJoCoEnv-v0"
-a    --agent         to choose which agent to run.
-t    --train         if set to True, the agent is also trained before evaluation.
-exe  --executor      select an execution model. By default the BaseExecutor is used which executes the action given from the agent without modification.
-obs  --observer      select an observer model. By default the baseObserver is used which passes the environment state as is to the agent.
-l    --logging       select logging level. "info" for  basic output; "debug" for debugging purposes.
-s    --seed          set the random seed that will ne used for numpy and PyTorch.
```

---

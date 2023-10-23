# Maximum State Entropy Exploration using Predecessor and Successor Representations
####  [[Project Website]]() [[Paper]](https://arnavkj1995.github.io/pubs/Jain23.pdf) [[Video]]()
Implementation of $\eta\psi$-Learning agent in PyTorch. If you find this code useful, please reference in your paper:
```
@inproceedings{
    jain2023maximum,
    title={Maximum State Entropy Exploration using Predecessor and Successor Representations},
    author={Jain, Arnav Kumar and Lehnert, Lucas and Rish, Irina and Berseth, Glen},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=tFsxtqGmkn}
}
```

## Abstract
Animals have a developed ability to explore that aids them in important tasks such as locating food, exploring for shelter, and finding misplaced items. These exploration skills necessarily track where they have been so that they can plan for finding items with relative efficiency. Contemporary exploration algorithms often
learn a less efficient exploration strategy because they either condition only on the current state or simply rely on making random open-loop exploratory moves. In this work, we propose $\eta\psi$-Learning, a method to learn efficient exploratory
policies by conditioning on past episodic experience to make the next exploratory move. Specifically, $\eta\psi$-Learning learns an exploration policy that maximizes the entropy of the state visitation distribution of a single trajectory. Furthermore, we
demonstrate how variants of the predecessor representation and successor representations can be combined to predict the state visitation entropy. Our experiments demonstrate the efficacy of $\eta\psi$-Learning to strategically explore the environment
and maximize the state coverage with limited samples.

## Running
This repository uses ``Python 3.7``. The dependencies can be installed using the ``requirements.txt`` file.

For running experiments, use the ``run.sh`` file name of the environment (``$ENV_NAME``) as arguments. 
```
bash run.sh $ENV_NAME
```

## Environments
Table providing the name of the environments used for experiments, if the action space is finite or infinite, and the description. 
| ENV_NAME       | Action Space | Description                    |
|------------|--------------|---------------------------------|
| chain | Finite | Basic ChainMDP with 6 states where agents starts in the center        |
| riverswim | Finite | Stochastic chainMDP where the agent starts in the center |
| grid       | Finite | NxN sized gridworld |
| tworooms      | Finite | Gridworld with two rooms and agent starts in the center of the rooms |
| fourrooms     | Finite | Gridworld with four rooms |
| reacher       | Infinite | Continuous control environment with two-jointed robotic arm  |
| pusher        | Infinite | Continuous control environment having a multiple-jointed robotic arm |
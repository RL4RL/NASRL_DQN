# NASRL_DQN
## NASRL_DQN is a PyTorch combination implementation of [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578.pdf) and [Deep Q learning](https://www.nature.com/articles/nature14236)

## Installation
```bash
git clone https://github.com/RL4RL/NASRL_DQN.git  
```
```python
pip install -r requirements.txt
```
Install [Pytorch](https://pytorch.org/)(v1.7)  
Add NASRL_DQN path into Python path: export PYTHONPATH=PATH_TO_NASRL_DQN  

## Running
### Train a single neural network on Atari datsets(Currently only for Pong game).
```python
python dql/dqn.py --conf block_conf.json
```

### Run Neural Architecture Search for multiple deep Q learning(Under construction)

```python
python main.py --game pong
```

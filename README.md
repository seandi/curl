# Reproduction of CURL+SAC for the DMControl experiments
This repository contains an implementation of three agents, SAC,SAC+AE and CURL with SAC, used to reproduce the experiments run in the original CULR paper[[1]](#1) for the DMControl100k and DMControl500k experiments, as part of a reproducibility analysis.The implemetations of the three agents follows those available at [[2]](#2) and [[3]](#3).

## Run instructions
A `curl_sac` agent can be trained on the `cartpole swingup` task for 100k environment steps with the following command
```
python main.py --agent=curl_sac --domain=cartpole --task=swingup --action_repeat=8 --env_steps_training=100000
```
The available agents are `sac`,`sacae` and `curl_sac`. Refer to the `parse_args.py` inside the utils folder for the list of available arguments and to [[1]](#1) for the complete list of hyper-parameters. 

## References
<a id="1">[1]</a> 
Aravind Srinivas and Michael Laskin and Pieter Abbeel.
CURL: Contrastive Unsupervised Representations for Reinforcement Learning. 2020.
https://arxiv.org/abs/2004.04136

<a id="2">[2]</a> 
CURL+SAC original implementation.
https://github.com/MishaLaskin/curl

<a id="3">[3]</a> 
SAC+AE original implementation.
https://github.com/denisyarats/pytorch_sac_ae

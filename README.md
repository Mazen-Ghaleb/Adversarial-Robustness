# Adversarial Robustness

A framework to test the effectiveness of adversarial attacks and defense mechanisms on autonomous driving systems. This framework is built on top of [CARLA](https://carla.org/) and [PyTorch](https://pytorch.org/).

## Demonstration
[![Watch the demonstration video](https://img.youtube.com/vi/OqU7b2n0Fk8/maxresdefault.jpg)](https://youtu.be/OqU7b2n0Fk8)

## How to install
1. Download [CARLA version 0.9.12](https://carla.org/2021/08/02/release-0.9.12/)
2. Clone the repository
3. Create carlaPath file in the carla folder, where it contains a carlaPath variable that has a path to the CARLA 0.9.12 folder.
4. Running install.sh
5. Setup local environment and downloading the dependencies from environment.yaml
6. (Optional: Step to enable CUDA, must have NVIDIA GPU)
Download CUDA 11.7 and PyTorch that is compatible with CUDA 11.7

## How to run
1. Running the CARLA 0.9.12 environment  
2. Running the script main.py in carla folder in the local environment by running the following commands
```
cd carla
python main.py
```
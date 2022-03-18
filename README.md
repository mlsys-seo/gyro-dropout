# Gyro Dropout: Maximizing Ensemble Effect in Neural Network Training
Gyro droput is a variant of dropout that improves the efficiency of training neural networks. Instead of randomly dropping out neurons in every training iteration, Gyro droput pre-selects and trains a fixed number of subnetworks. Because each subnetwork is more stably trained, they are more diversified and thus their ensemble achieves good generalization. Gyro dropout improves the accuracy of trained models by up to 1.93%; gyro dropout consistently achieves higher accuracy than conventional dropout in all experiments.

## Prerequisites
- Environment setting
    - NVIDIA Titan V100(32GB) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz or NVIDIA Titan XP(12GB) with Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
    - Tensorflow v1.12
    - CUDA 9

- Available models/datasets in the code
    - Models
        1. AlexNet
        2. LeNet
        3. 4 x FC MLP
    - Datasets
        1. CIFAR10
        2. CIFAR100

## Instruction for running experiments
1. Run the following command for running the experiment without dropout.
```bash
python main.py --model=lenet --dataset=cifar100 --do_augmentation=True
```
2. Run the following command for running the experiment with conventional dropout.
```bash
python main.py --model=lenet --dataset=cifar100 --do_conventional_dropout=True --do_augmentation=True
```
3. Run the following command for running the experiment with gyro dropout.
```bash
python main.py --model=lenet --dataset=cifar100 --do_gyro_dropout=True --do_augmentation=True
```

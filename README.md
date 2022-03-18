# Gyro Dropout: Maximizing Ensemble Effect in Neural Network Training

- Environment setting: NVIDIA Titan V100(32GB) with Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz or NVIDIA Titan XP(12GB) with Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz

- Available models/datasets in the code
    - Models
        1. AlexNet
        2. LeNet
        3. 4 x FC MLP
    - Datasets
        1. CIFAR10
        2. CIFAR100


## Instruction for running no dropout experiment
1. Run the following command for running the experiment.
```bash
python main.py --model=lenet --dataset=cifar100 --do_augmentation=True
```


## Instruction for running conventional dropout experiment
1. Run the following command for running the experiment.
```bash
python main.py --model=lenet --dataset=cifar100 --do_gyro_dropout=True --do_augmentation=True
```


## Instruction for running gyro dropout experiment
1. Run the following command for running the experiment.
```bash
python main.py --model=lenet --dataset=cifar100 --do_conventional_dropout=True --do_augmentation=True
```


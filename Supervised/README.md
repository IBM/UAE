# Supervised Attack
Code for MINE-based attack of MinMax and Binary Search algorithms

# Software Version
The program is developed under Tensorflow-gpu 1.12.0. Note that we use Keras embedded inside Tensorflow.

# Dataset
The dataset **MNIST** and **CIFAR-10** will be downloaded automatically when you use them first time.

# Classifiers
To obtain the classifiers for **MNIST** and **CIFAR-10** run
```
python train_model.py
```
# Run Attacks
To attack the classifier, run
```
python test_attack.py
```
We show the detail of the attack as following.

## Binary Search Algorithm

```python
from MINE_binary_search import MINE_supervised_binary_search

MINE_supervised_binary_search(sess, model, batch_size=1, max_iterations=1000, confidence=0,
            targeted=False, epsilon=1.0, mine_batch='conv').attack(inputs, targets)
```

## MinMax Algorithm

```python
from supervised_attack import MINE_supervised

MINE_supervised(sess, model,batch_size=1, max_iterations=1000, confidence=0, 
            targeted=False, epsilon=1.0, mine_batch='conv').attack(inputs, targets)
```
where inputs are a tensor (batch, height, weight, channel) and targets are a tensor (batch, classes). In MINE-based attack ($L_\infty$ attack), we only can use ```batch_size=1``` and set perturbation level $\epsilon$ by ```epsilon=XX```. To use MINE, you can generate a batch samples by convolution output with using ```mine_batch = 'conv'``` or generate a batch by random sampling with ```mine_batch='random_sampling'```. It runs untargeted attack by setting```target=False```.


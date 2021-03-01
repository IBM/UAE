# Unsupervised attack
Code for MINE-based unsupervised attack of MinMax algorithm.

# Software Version
The program is developed under Tensorflow-gpu 1.12.0. Note that we use Keras embedded inside Tensorflow.

# Dataset
For UAE improving Data Reconstruction, the dataset **MNIST**, **FASHION MNIST** and **SVHN** will be downloaded automatically when you use them first time.
Fou UAE improving Representation Learning, we put the dataset, **Mice Protein**, **Activity**, in folder ```data/ ```.
You can download dataset **Isolet** with the link: https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/ and download dataset **COIL-20** (processed) by https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php.

# Autoencoder
We provide 5 autoencoders and put the code in folder ```Autoencoders/```. The example for training the convolution autoencoder, run
```
python convAE.py
```
## Concrete-Autoendoer
To inference Concrete-Autoencoder, you need to run
```
pip install concrete-autoencoder
```

# Run Attacks
To conduct MINE-based unsupervised attack, run
```
python test_attack.py
```
The detail of attack, we show as following.

## MinMax Algorithm
```python
from unsupervised_attack import MINE_unsupervised
MINE_unsupervised(sess, model,batch_size=1, max_iterations=40, 
    epsilon=1.0, mine_batch='conv').attack(inputs)
```
where inputs are a tensor of training data. In MINE-based attack (L_$\infty$ attack), we only can use batch_size=1 and set perturbation level $\epsilon$ by epsilon=XX. To use MINE, you can generate a batch samples by convolution output with using ```mine_batch = 'conv'``` or generate a batch by random sampling with ```mine_batch='random_sampling'```.

#  Contrastive Learning
The detail of Contrastive Learning is shown in folder ```simCLR/```. 

# Contrastive Learning
Code for MINE-based unsupervised of the MinMax algorithm.
We use code of "A Simple Framework for Contrastive Learning of Visual Representations" with the link: https://github.com/google-research/simclr.

# Dataset
The dataset **CIFAR-10** will be downloaded automatically when you use them first time.

# Classifier
Using code from https://github.com/google-research/simclr, it runs
```python
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False
  
  python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=/tmp/simclr_test --model_dir=/tmp/simclr_test_ft --use_tpu=False

```

# Run Attacks
To conduct unsupervised attack, run
```
python test_attack.py
```
The detail of the attack is shown as following.

## MinMax Algorithm
```python
from contrast_attack import MINE_MinMax_unsupervised

MINE_MinMax_unsupervised(sess, model, batch_size=1, 
    max_iterations=20, epsilon=1.0).attack(inputs, targets)
```
where inputs are a tensor of training data. Although the input of attack contains targets, it doesn't use labels for the contrastive loss. In MINE-based attack ($L_\infty$ attack), we only can use batch_size=1 and set perturbation level $\epsilon$ by epsilon=XX.

# Re-training the Classifier with UAE
To re-train the classifier with UAE, run
```python
python run_uae.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1500 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False --uae_path=./uae/uae.npy
  
python run_uae.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=/tmp/simclr_test --model_dir=/tmp/simclr_test_ft --use_tpu=False --uae_path=./uae/uae.npy

```
We need to load UAE with ```--uae_path```.

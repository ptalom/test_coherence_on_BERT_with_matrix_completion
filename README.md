
## Test Coherence on BERT, case study on Matrix Completion
In the original paper "Abrupt Learning in Transformers: A Case Study on Matrix Completion" (NeurIPS 2024). [arXiv link](https://arxiv.org/abs/2410.22244), the aim is to train a model on a matrix completion task. To do this, the authors generate matrix, then randomly select positions of the elements of these matrix with a `p_mask` proportion, and mask them before sending them to the model, which will attempt to predict them. 

For the case of this mini-project, I want to test the model's sensitivity when the masked positions are chosen according to a certain coherence and not randomly as in the original paper. To do this, I calculate the local coherence for each of the matrices that is generated and I select the positions to mask based only on coherence. I use τ as a coherence controller so that, for example, if τ = 1 I take all (100%) the entries that have strong local coherence and apply the `p_mask` on them, if τ = 0.5, I keep only half of the strong coherence values ​​and apply the `p_mask` proportion on these positions.

## Setup 
```bash
git clone https://github.com/ptalom/test_coherence_on_BERT_with_matrix_completion.git
cd test_coherence_on_BERT_with_matrix_completion
cd src
```

## Start training
```bash
python train.py --config configs/train.yaml
```

## Contributor
- Patrick C. Talom
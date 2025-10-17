
## Test Coherence on BERT, case study on Matrix Completion
In the original paper "Abrupt Learning in Transformers: A Case Study on Matrix Completion" (NeurIPS 2024). [arXiv link](https://arxiv.org/abs/2410.22244), the aim is to train a model on a matrix completion task. To do this, the authors generate matrix, then randomly select positions of the elements of these matrix with a `p_mask` proportion, and mask them before sending them to the model, which will attempt to predict them. 

For this mini-project, I want to test the model's sensitivity when the masked positions are selected according to a coherence-based criterion rather than randomly, as in the original paper. To achieve this, I compute the local coherence for each generated matrix and select the positions to mask based on this coherence.

I use τ as a coherence controller:

- If τ = 1, I select all the positions with the highest local coherence and apply the `p_mask` proportion on these positions.

- If τ = 0.5, I select only the top 50% of positions according to their local coherence and apply the `p_mask` proportion on these selected positions.

- If τ = 0, no coherence-based selection is done, and the masked positions are chosen entirely at random according to `p_mask`.

This ensures that exactly `p_mask * (m * n)` entries are masked in each matrix, with a fraction τ determined by coherence, while the remaining `(1-τ) * N` entries (if τ < 1) can be chosen uniformly at random.

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
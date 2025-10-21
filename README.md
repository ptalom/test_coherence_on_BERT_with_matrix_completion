
## Test Coherence on BERT, case study on Matrix Completion
In the original paper "Abrupt Learning in Transformers: A Case Study on Matrix Completion" (NeurIPS 2024). [arXiv link](https://arxiv.org/abs/2410.22244), the aim is to train a model on a matrix completion task. To do this, the authors generate matrix, then randomly select positions of the elements of these matrix with a `p_mask` proportion, and mask them before sending them to the model, which will attempt to predict them. 

For this mini-project, I want to test the model's sensitivity when the masked positions are selected according to a coherence-based criterion rather than randomly, as in the original paper. To achieve this, I compute the local coherence for each generated matrix and select the positions to mask based on this coherence.

I use τ as a coherence controller:

- If τ = 1, I select all the positions with the highest local coherence and apply the `p_mask` proportion on these positions. 

- If τ = 0.5, I select the top 50% of positions according to their local coherence and apply the `p_mask` proportion on these selected positions and I complete the remaining 50% by masking the positions randomly.

- If τ = 0, no coherence-based selection is done, and the masked positions are chosen entirely at random according to `p_mask`.

This ensures that exactly `p_mask * (m * n)` entries are masked in each matrix, with a fraction τ determined by coherence, while the remaining `(1-τ) * N` entries (if τ < 1) can be chosen uniformly at random.

### Experimental setup
- Model: BERT trained for matrix completion (as in the original paper) with the same parameters 

### Results
The results show that, the masking strategy has a strong impact on model performance:
- When **τ = 1 (fully coherence-based masking)** as shown in [Fig 1], the model performance decreases significantly compared to uniform masking (τ = 0, [Fig 2]). The sudden drop in loss that is present in the original paper (τ = 0) is completly absent under fully coherent masking. For intermediate coherence (τ = 0.5, [Fig 3]), the sudden drop is partially recovered, although it is still weaker than in the uniform masking case.

- **Role of randomness**: These observations suggest that the Transformer relies on a certain degree of randomness in the masked entries to generalize effectively.

- **Bias induced by coherence-based masking**: Uniform masking appears to promote more robust learning, while coherence-focused masking introduces strong sampling bias, which reduces recovery performance.

| ![Fig 1](images/training_tau_1.png) | ![Fig 2](images/training_tau_0.png) | ![Fig 3](images/training_tau_05.png) |
|:--:|:--:|:--:|
| *Fig. 1 – Coherence-based masking (tau=1)* | *Fig. 2 – Coherence-based masking (tau=0)* | *Fig. 3 – Coherence-based masking (tau=0.5)* |

### Conclusion
- This experiment indicates that BERT (Transformers) trained for matrix completion are sensitive to structured masking patterns.
- High-coherence masking (τ = 1) hinders learning, whereas random masking (τ = 0) preserves generalization.

[Fig 4] and [Fig 5] respectively show on the same image the evolution of the error according to tau during the training
| ![Fig 4](images/train_loss_all_tau.png) | ![Fig 5](images/mask_loss_all_tau.png) |
|:--:|:--:|
| *Fig. 4 – Train loss (L) evolution* | *Fig. 5 – Mask loss (L_mask) evolution* |


## Additional Experiment — Coherence based in Convex Methods
I also tested the impact of local coherence on convex approaches for matrix completion, in particular Nuclear Norm Minimization

### Experimental setup
- Parameters:
    - Masking proportion `p_mask` ∈ `{0.1, 0.3, 0.5, 0.7, 0.9}`
    - Coherence control parameter τ ∈ [0, 1]
- Metric: Mean Squared Error (MSE) on observed, masked, and total entries.

For each configuration, I ran the solver multiple times with different random seeds and plotted the mean ± standard deviation to ensure consistent and robust comparisons

### Results
[Fig 6] shows the effect of coherence-based masking on the reconstruction performance of the convex low-rank solver
1. Total reconstruction error (`L_mean`) and masked-entry error (`L_mask_mean`) increase with τ.  
2. Error on observed entries (`L_obs_mean`) remains nearly constant.  
3. Low masking rates (`p_mask ≤ 0.3`) tolerate coherence reasonably well, while higher masking rates (`p_mask ≥ 0.5`) exhibit a sharp degradation when τ → 1.

These results confirm that, coherence-based masking significantly reduces recovery performance, especially for large fractions of missing entries, demonstrating the sensitivity of convex low-rank recovery methods to mask structure.

 ![Fig 6](images/cvx.png)
 *Fig. 6 – MSE evolution according to tau for different p_mask* 

## To reproduce this work
### Setup 
```bash
git clone https://github.com/ptalom/test_coherence_on_BERT_with_matrix_completion.git
cd test_coherence_on_BERT_with_matrix_completion
cd src
```

### Getting started
Install the dependencies using Conda
```bash
conda env create -f env.yaml
conda activate coherence_sampling
```

### Training
```bash
python train.py --config configs/train.yaml
```

### Contributor
- Patrick C. Talom
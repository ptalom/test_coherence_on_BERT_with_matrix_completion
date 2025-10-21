import torch
from abc import ABC, abstractmethod
import numpy as np

class Matrix(ABC):
    def __init__(self, min_val, max_val, min_vocab, max_vocab):
        self.min_val = min_val
        self.max_val = max_val
        self.min_vocab = min_vocab
        self.max_vocab = max_vocab
        self.vocab = {"MASK": 0}

    @abstractmethod
    def construct_vocab(self):
        pass

    @abstractmethod
    def tokenize(self, X, mask):
        pass

    @abstractmethod
    def sample(self, n_samples, m, n, r, p_mask, tau, return_uv=False):
        pass


class RealMatrix(Matrix):
    def __init__(self, args, device):
        super().__init__(
            min_val=args.min_val,
            max_val=args.max_val,
            min_vocab=args.min_vocab,
            max_vocab=args.max_vocab,
        )
        self.prec = args.prec
        self.device = device
        self.construct_vocab()

    def construct_vocab(self):
        for val in range(
            int(self.min_vocab * (10**self.prec)),
            int((self.max_vocab * 10**self.prec) + 1),
        ):
            self.vocab[str(val / (10**self.prec))] = len(self.vocab.keys())

    def tokenize(self, X, mask):
        X_rounded = torch.round(X, decimals=self.prec)
        X_mask_token = (
            1 + ((X_rounded * 10**self.prec) - (self.min_vocab * 10**self.prec))
        ).to(torch.int) * mask
        return X_mask_token.view(X.shape[0], -1), X_rounded.to(torch.float)

    @staticmethod
    def calculate_local_coherence(A):
        """Calcule la cohérence locale via SVD"""
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        r = S.shape[1]
        mu = (A.shape[1] / r) * torch.sum(U**2, dim=2)
        nu = (A.shape[2] / r) * torch.sum(Vh**2, dim=1)
        return mu[:, :, None] + nu[:, None, :]

    def sample(self, n_samples, m, n, r, p_mask, tau, return_uv=False):
        """
        Génère une matrice et masque N = p_mask * m * n éléments.
        Parmi ces N :
            - tau * N sont choisis selon la cohérence locale
            - (1 - tau) * N sont choisis uniformément
        """

        U = self.min_val + (self.max_val - self.min_val) * torch.rand(
            (n_samples, m, r), device=self.device
        )
        V = self.min_val + (self.max_val - self.min_val) * torch.rand(
            (n_samples, n, r), device=self.device
        )
        matrix = U @ V.permute(0, 2, 1)

        batch_size, _, _ = matrix.shape
        total = m * n
        N = int(p_mask * total)
        N_tau = int(tau * N)

        if tau > 0:
            P = self.calculate_local_coherence(matrix)
            P_flat = P.reshape(batch_size, -1)
            #les plus cohérentes
            sorted_idx = torch.argsort(P_flat, dim=1, descending=True)
            
            #les moins cohérentes
            #sorted_idx = torch.argsort(P_flat, dim=1, descending=False)

            coh_mask_flat = torch.zeros_like(P_flat, dtype=torch.bool)
            coh_mask_flat.scatter_(1, sorted_idx[:, :N_tau], True)
            coh_mask = coh_mask_flat.view(batch_size, m, n)
        else:
            coh_mask = torch.zeros((batch_size, m, n), dtype=torch.bool, device=self.device)

        rand_flat = torch.rand((batch_size, total), device=self.device)
        rand_sorted_idx = torch.argsort(rand_flat, dim=1)
        rand_mask_flat = torch.zeros_like(rand_flat, dtype=torch.bool)
        rand_mask_flat.scatter_(1, rand_sorted_idx[:, : (N - N_tau)], True)
        rand_mask = rand_mask_flat.view(batch_size, m, n)
        rand_mask = rand_mask & (~coh_mask)

        mask_total = coh_mask | rand_mask
        visible_mask = (~mask_total).to(torch.int)  # 1 = visible, 0 = masqué

        
        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, visible_mask)

        if return_uv:
            return matrix_mask_tok, matrix_rounded, visible_mask, U, V
        return matrix_mask_tok, matrix_rounded, visible_mask

    def sample_patch(self, n_samples, m, n, r, p_mask, tau, return_uv=False):
        """
        Variante avec matrice corrigée corr_matrix = -matrix
        """
        matrix_mask_tok, matrix_rounded, mask, U, V = self.sample(
            n_samples, m, n, r, p_mask, tau, return_uv=True
        )

        corr_matrix = -1 * (U @ V.permute(0, 2, 1))
        corr_mask_tok, corr_matrix_rounded = self.tokenize(corr_matrix, mask)

        return (
            matrix_mask_tok.to(self.device),
            matrix_rounded.to(self.device),
            corr_mask_tok.to(self.device),
            corr_matrix_rounded.to(self.device),
            mask.to(self.device),
        )


class GaussianMatrix(RealMatrix):
    def __init__(self, args, device):
        super(GaussianMatrix, self).__init__(args=args, device=device)
        self.scale = args.gaussian_scale

    def sample(self, n_samples, m, n, r, p_mask, tau, return_uv=False):
        print("Gaussian")
        U = self.scale * torch.randn((n_samples, m, r), device=self.device)
        V = self.scale * torch.randn((n_samples, n, r), device=self.device)
        matrix = U @ V.permute(0, 2, 1)
        return super().sample(n_samples, m, n, r, p_mask, tau, return_uv)


class LaplaceMatrix(RealMatrix):
    def __init__(self, args, device):
        super(LaplaceMatrix, self).__init__(args=args, device=device)
        self.scale = args.laplace_scale

    def sample(self, n_samples, m, n, r, p_mask, tau, return_uv=False):
        print("Laplace")
        laplace = torch.distributions.laplace.Laplace(0, self.scale)
        U = laplace.sample((n_samples, m, r)).to(self.device)
        V = laplace.sample((n_samples, n, r)).to(self.device)
        matrix = U @ V.permute(0, 2, 1)
        return super().sample(n_samples, m, n, r, p_mask, tau, return_uv)

import torch
from abc import ABC, abstractmethod
import numpy as np
import math


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
        super(RealMatrix, self).__init__(
            min_val=args.min_val,
            max_val=args.max_val,
            min_vocab=args.min_vocab,
            max_vocab=args.max_vocab,
        )

        self.prec = args.prec
        self.construct_vocab()
        self.device = device

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

        X_mask_token = X_mask_token.view(X.shape[0], -1)
        return X_mask_token, X_rounded.to(torch.float)
    
    @staticmethod
    def calculate_local_coherence(A):
        """
        Calcule une mesure de cohérence locale basée sur la SVD.
        A : (batch_size, m, n)
        """
        batch_size, m, n = A.shape
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        r = S.shape[1]
        mu = (m / r) * torch.sum(U**2, dim=2)  # (batch, m)
        nu = (n / r) * torch.sum(Vh**2, dim=1)  # (batch, n)
        P = mu[:, :, None] + nu[:, None, :]  # (batch, m, n)
        return P

    def sample(self, n_samples, m, n, r, p_mask, tau=1.0, return_uv=False):
        """
        - Génère la matrice.
        - Calcule la cohérence.
        - Sélectionne les tau% des positions les plus cohérentes.
        - Applique p_mask uniquement sur ces positions.
        """
        # --- Génération de la matrice ---
        U = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, m, r), device=self.device
        )

        V = self.min_val + (self.max_val - self.min_val) * torch.rand(
            size=(n_samples, n, r), device=self.device
        )

        matrix = U @ V.permute(0, 2, 1)

        # --- Calcul cohérence ---
        P = self.calculate_local_coherence(matrix)
        batch_size = P.shape[0]
        N = m * n
        N_tau = int(tau * N)  # nb de positions cohérentes à retenir

        # --- Création du masque basé sur cohérence ---
        P_flat = P.reshape(batch_size, -1)
        sorted_indices = torch.argsort(P_flat, dim=1, descending=True)

        mask_batch = torch.zeros_like(P_flat, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            # positions les plus cohérentes
            coh_indices = sorted_indices[b, :N_tau]
            mask_batch[b, coh_indices] = True

        # reshape du masque cohérent
        coherence_mask = mask_batch.view(batch_size, m, n)

        # --- Application du p_mask sur ces positions cohérentes ---
        rand_mask = torch.rand_like(matrix, device=self.device)
        visible_mask = (rand_mask > p_mask) & coherence_mask  # on garde seulement p_mask sur cohérentes

        # --- Tokenisation ---
        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, visible_mask)

        if return_uv:
            return matrix_mask_tok, matrix_rounded, visible_mask, U, V

        return matrix_mask_tok, matrix_rounded, visible_mask


class GaussianMatrix(RealMatrix):
    def __init__(self, args, device):
        super(GaussianMatrix, self).__init__(args=args, device=device)
        self.scale = args.gaussian_scale

    def sample(self, n_samples, m, n, r, p_mask, tau=1.0, return_uv=False):
        print("Gaussian")
        U = self.scale * torch.randn((n_samples, m, r), device=self.device)
        V = self.scale * torch.randn((n_samples, n, r), device=self.device)
        matrix = U @ V.permute(0, 2, 1)

        P = self.calculate_local_coherence(matrix)
        batch_size = P.shape[0]
        N = m * n
        N_tau = int(tau * N)

        P_flat = P.reshape(batch_size, -1)
        sorted_indices = torch.argsort(P_flat, dim=1, descending=True)
        mask_batch = torch.zeros_like(P_flat, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            coh_indices = sorted_indices[b, :N_tau]
            mask_batch[b, coh_indices] = True

        coherence_mask = mask_batch.view(batch_size, m, n)
        rand_mask = torch.rand_like(matrix, device=self.device)
        visible_mask = (rand_mask > p_mask) & coherence_mask

        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, visible_mask)
        if return_uv:
            return matrix_mask_tok, matrix_rounded, visible_mask, U, V
        return matrix_mask_tok, matrix_rounded, visible_mask


class LaplaceMatrix(RealMatrix):
    def __init__(self, args, device):
        super(LaplaceMatrix, self).__init__(args=args, device=device)
        self.scale = args.laplace_scale

    def sample(self, n_samples, m, n, r, p_mask, tau=1.0, return_uv=False):
        print("Laplace")
        laplace = torch.distributions.laplace.Laplace(loc=0, scale=self.scale)
        U = laplace.sample((n_samples, m, r)).to(self.device)
        V = laplace.sample((n_samples, n, r)).to(self.device)
        matrix = U @ V.permute(0, 2, 1)

        P = self.calculate_local_coherence(matrix)
        batch_size = P.shape[0]
        N = m * n
        N_tau = int(tau * N)

        P_flat = P.reshape(batch_size, -1)
        sorted_indices = torch.argsort(P_flat, dim=1, descending=True)
        mask_batch = torch.zeros_like(P_flat, dtype=torch.bool, device=self.device)

        for b in range(batch_size):
            coh_indices = sorted_indices[b, :N_tau]
            mask_batch[b, coh_indices] = True

        coherence_mask = mask_batch.view(batch_size, m, n)
        rand_mask = torch.rand_like(matrix, device=self.device)
        visible_mask = (rand_mask > p_mask) & coherence_mask

        matrix_mask_tok, matrix_rounded = self.tokenize(matrix, visible_mask)
        if return_uv:
            return matrix_mask_tok, matrix_rounded, visible_mask, U, V
        return matrix_mask_tok, matrix_rounded, visible_mask

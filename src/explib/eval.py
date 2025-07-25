import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, binned_statistic
from sklearn.metrics import mutual_info_score
from torch.distributions import MultivariateNormal

class RadialFlowEvaluator:
    def __init__(self, flow, data, device='cpu'):
        """
        Evaluator for USFlow models with RadialDistribution base distribution.
        
        Args:
            flow: Trained USFlow model with RadialDistribution base
            data: Dataset tensor for evaluation
            device: Device for computation
        """
        self.flow = flow.to(device)
        self.data = data.to(device)
        self.device = device
        
        self.dim = torch.prod(torch.tensor(self.data.shape[1:])).item()

        # Precompute latent representations
        with torch.no_grad():
            self.latents = self.flow.backward(self.data) - flow.base_distribution.loc.to(device)
            self.latents = self.latents.view(self.latents.shape[0], -1)
        # Get p-norm from base distribution
        self.p = flow.base_distribution.p
    
    def wasserstein_norm_distance(self, n_samples=10000):
        """
        Compute Wasserstein distance between:
        1. Norm distribution of base distribution
        2. Empirical p-norms of latent representations
        
        Returns:
            wasserstein_dist: Wasserstein distance
        """
        # Get empirical latent norms
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        
        # Sample from base norm distribution
        base_norm_dist = self.flow.base_distribution.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        # Compute Wasserstein distance
        return wasserstein_distance(latent_norms, sample_norms)
    
    def ks_norm_statistic(self, n_samples=10000):
        """
        Compute Kolmogorov-Smirnov statistic for norm distributions.
        
        Returns:
            ks_stat: KS statistic
            p_value: Associated p-value
        """
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        base_norm_dist = self.flow.base_distribution.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        return ks_2samp(latent_norms, sample_norms)
    
    def qq_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate QQ-plot comparing:
        1. Quantiles of empirical latent norms
        2. Quantiles of base norm distribution samples
        """
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        base_norm_dist = self.flow.base_distribution.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        latent_quantiles = np.quantile(latent_norms, np.linspace(0, 1, 100))
        sample_quantiles = np.quantile(sample_norms, np.linspace(0, 1, 100))
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(sample_quantiles, latent_quantiles, alpha=0.7)
        min_val = min(sample_quantiles.min(), latent_quantiles.min())
        max_val = max(sample_quantiles.max(), latent_quantiles.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title('QQ-plot of Latent Norms')
        ax.set_xlabel('Base Distribution Quantiles')
        ax.set_ylabel('Data Latent Quantiles')
        return ax
    
    def kde_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate KDE plots comparing:
        1. Empirical latent norms distribution
        2. Base norm distribution
        """
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        base_norm_dist = self.flow.base_distribution.norm_distribution
        sample_norms = base_norm_dist.sample((n_samples,)).cpu().numpy()
        
        # Create KDE models
        kde_latent = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(latent_norms.reshape(-1, 1))
        kde_base = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(sample_norms.reshape(-1, 1))
        
        # Evaluate on grid
        x_grid = np.linspace(
            min(latent_norms.min(), sample_norms.min()),
            max(latent_norms.max(), sample_norms.max()),
            1000
        )[:, np.newaxis]
        
        log_dens_latent = kde_latent.score_samples(x_grid)
        log_dens_base = kde_base.score_samples(x_grid)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_grid, np.exp(log_dens_latent), label='Data Latents')
        ax.plot(x_grid, np.exp(log_dens_base), label='Base Distribution')
        ax.set_title('KDE of Norm Distributions')
        ax.set_xlabel('Norm Value')
        ax.set_ylabel('Density')
        ax.legend()
        return ax
    
    def pp_plot_norms(self, ax=None, n_samples=10000):
        """
        Generate PP-plot comparing:
        1. Empirical CDF of latent norms
        2. Theoretical CDF of base norm distribution
        
        Args:
            ax: Matplotlib axis (optional)
            n_samples: Number of samples for theoretical distribution
            
        Returns:
            ax: Matplotlib axis
        """
        # Get empirical latent norms
        latent_norms = torch.norm(self.latents, p=self.p, dim=1).cpu().numpy()
        
        # Compute empirical CDF
        n = len(latent_norms)
        empirical_cdf = np.arange(1, n+1) / n
        sorted_norms = np.sort(latent_norms)
        
        # Get theoretical CDF (if available)
        base_norm_dist = self.flow.base_distribution.norm_distribution
        if hasattr(base_norm_dist.distribution, 'cdf'):
            # Use analytical CDF if available
            theoretical_cdf = base_norm_dist.distribution.cdf(
                torch.tensor(sorted_norms).to(self.device)
            ).detach().cpu().numpy()
        else:
            # Approximate via sampling
            sample_norms = base_norm_dist.sample((n_samples,)).detach().cpu().numpy()
            sample_sorted = np.sort(sample_norms)
            theoretical_cdf = np.searchsorted(sample_sorted, sorted_norms) / n_samples
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(theoretical_cdf, empirical_cdf, alpha=0.7)
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_title('PP-plot of Norm Distributions')
        ax.set_xlabel('Theoretical CDF (Base Distribution)')
        ax.set_ylabel('Empirical CDF (Data Latents)')
        ax.grid(True)
        
        return ax
    
    def binned_uniformity_test(self, n_bins=10):
        """
        Binned uniformity test for latent directions.
        Computes chi-squared statistic for binned directional data.
        
        Returns:
            chi2_stat: Chi-squared statistic
            p_value: Associated p-value
        """
        # Normalize to unit sphere
        directions = self.latents / torch.norm(self.latents, p=self.p, dim=1, keepdim=True)
        directions = directions.cpu().numpy()
        
        # Create bins in each dimension
        bin_edges = np.linspace(-1, 1, n_bins + 1)
        bin_indices = np.zeros(len(directions), dtype=int)
        
        # Multi-dimensional binning
        for dim in range(self.dim):
            bin_indices_dim = np.digitize(directions[:, dim], bin_edges) - 1
            bin_indices += bin_indices_dim * (n_bins ** dim)
        
        # Count bins
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        n_observed = len(unique_bins)
        
        # Expected counts (uniform distribution)
        total_bins = n_bins ** self.dim
        expected = len(directions) / total_bins
        
        # Chi-squared test
        chi2_stat = np.sum((counts - expected) ** 2 / expected)
        p_value = 1 - chi2.cdf(chi2_stat, df=n_observed - 1)
        
        return chi2_stat, p_value
    
    def hs_independence_test(self, n_permutations=1000):
        """
        Hilbert-Schmidt Independence Criterion for:
        H0: Norm and direction are independent
        
        Returns:
            hsic_value: HSIC statistic
            p_value: Estimated p-value via permutation test
        """
        # Compute norms and directions
        norms = torch.norm(self.latents, p=self.p, dim=1).unsqueeze(1)
        directions = self.latents / norms
        
        # Center and scale
        norms = (norms - norms.mean()) / norms.std()
        directions = (directions - directions.mean(dim=0)) / directions.std(dim=0)
        
        # Compute kernels
        K_n = self._rbf_kernel(norms)
        K_d = self._rbf_kernel(directions)
        
        # Center kernels
        n = len(norms)
        H = torch.eye(n) - torch.ones(n, n) / n
        K_n = H @ K_n @ H
        K_d = H @ K_d @ H
        
        # Compute HSIC
        hsic_value = torch.trace(K_n @ K_d) / (n * n)
        
        # Permutation test for p-value
        permuted_values = []
        for _ in range(n_permutations):
            perm_idx = torch.randperm(n)
            K_d_perm = K_d[perm_idx][:, perm_idx]
            permuted_values.append(torch.trace(K_n @ K_d_perm).item())
        
        permuted_values = np.array(permuted_values) / (n * n)
        p_value = (permuted_values >= hsic_value.item()).mean()
        
        return hsic_value.item(), p_value
    
    def _rbf_kernel(self, X, sigma=None):
        """Compute RBF kernel matrix"""
        n = X.shape[0]
        X_norm = torch.sum(X**2, dim=1).reshape(-1, 1)
        pairwise_dist = X_norm + X_norm.T - 2 * torch.mm(X, X.T)
        
        if sigma is None:
            sigma = torch.median(pairwise_dist[pairwise_dist > 0]).sqrt()
        
        gamma = 1.0 / (2 * sigma**2)
        K = torch.exp(-gamma * pairwise_dist)
        return K
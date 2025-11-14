import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# ----------------------------
# Helper functions (DDPM math)
# ----------------------------

def make_beta_schedule(beta_start: float, beta_end: float, T: int, device=None) -> torch.Tensor:
    """Linear beta schedule (T,)"""
    return torch.linspace(beta_start, beta_end, T, device=device)

def compute_alphas(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given betas (T,), compute:
      alphas = 1 - betas
      alphas_cumprod = \bar\alpha_t
      sqrt_alphas_cumprod and sqrt_1m_alphas_cumprod for convenience
    All returned have shape (T,)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    
    return alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod

def q_posterior_mean_variance(x_t: torch.Tensor,
                              x0: torch.Tensor,
                              t: int,
                              betas: torch.Tensor,
                              alphas: torch.Tensor,
                              alphas_cumprod: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Compute exact q(x_{t-1} | x_t, x0)'s mean (tilde_mu_t) and variance (tilde_beta_t).
    - x_t: (B, *data_shape)
    - x0: (B, *data_shape)
    - t: integer in [1..T-1] (indexing assumes betas[0] = beta_1)
    Returns:
      tilde_mu: (B, *data_shape)
      tilde_beta: scalar float (the variance)  [isotropic scalar for DDPM]
    """
    # indices in arrays are 0-based; beta_t is betas[t]
    beta_t = betas[t]
    alpha_t = alphas[t]
    
    alpha_prev = alphas[t-1] if t >= 1 else torch.tensor(1.0, device=betas.device)  # unused for t=0
    bar_alpha_t = alphas_cumprod[t]
    bar_alpha_prev = alphas_cumprod[t-1] if t >= 1 else torch.tensor(1.0, device=betas.device)

    # tilde beta
    tilde_beta_t = beta_t * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha_t)

    # coefficients for mean formula (from the DDPM paper)
    coef_x0 = (torch.sqrt(bar_alpha_prev) * beta_t) / (1.0 - bar_alpha_t)
    coef_xt = torch.sqrt(alpha_t) * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha_t)

    tilde_mu = coef_x0.view(1, *([1] * (x_t.dim() - 1))) * x0 + coef_xt.view(1, *([1] * (x_t.dim() - 1))) * x_t
    return tilde_mu, float(tilde_beta_t)

def gaussian_kl_diag(mu_q: torch.Tensor,
                     var_q: torch.Tensor,
                     mu_p: torch.Tensor,
                     var_p: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu_q, var_q I) || N(mu_p, var_p I)) computed per sample, summed over data dims.
    - mu_q, mu_p: (B, *data_shape)
    - var_q, var_p: either scalars or tensors shape (T,) or (B,1,1,...)
    Returns: kl: (B,) per-sample KL (scalar per example)
    """
    # flatten spatial dims, keep batch
    B = mu_q.shape[0]
    d = mu_q[0].numel()
    mu_q_flat = mu_q.view(B, -1)
    mu_p_flat = mu_p.view(B, -1)

    # allow var_q/var_p scalars or same-shaped tensors broadcastable to data dims
    # for simplicity convert to scalars if possible
    if torch.is_tensor(var_q) and var_q.numel() == 1:
        var_q = float(var_q.item())
    if torch.is_tensor(var_p) and var_p.numel() == 1:
        var_p = float(var_p.item())
    
    # If scalar isotropic variances:
    if isinstance(var_q, float) and isinstance(var_p, float):
        # KL = 0.5 * ( log(var_p/var_q) - d + (d*var_q/var_p) + (mu_p-mu_q)^2 / var_p ).sum_over_dims
        log_term = 0.5 * (d * math.log(var_p / var_q))
        trace_term = 0.5 * (d * (var_q / var_p) - d)
        diff_sq = torch.sum((mu_p_flat - mu_q_flat) ** 2, dim=1)
        quad = 0.5 * diff_sq / var_p
        kl = log_term + trace_term + quad  # shape (B,)
        return kl
    else:
        # more general (per-element diag variances) - not fully vectorized for arbitrary shapes,
        # but we support case var tensors broadcastable to mu shapes.
        # Expand to same flat shape
        var_q_full = torch.as_tensor(var_q, device=mu_q.device).expand_as(mu_q)
        var_p_full = torch.as_tensor(var_p, device=mu_q.device).expand_as(mu_q)
        var_q_flat = var_q_full.view(B, -1)
        var_p_flat = var_p_full.view(B, -1)

        log_terms = 0.5 * torch.sum(torch.log(var_p_flat) - torch.log(var_q_flat), dim=1)
        trace_terms = 0.5 * torch.sum(var_q_flat / var_p_flat - 1.0, dim=1)
        quad = 0.5 * torch.sum((mu_p_flat - mu_q_flat) ** 2 / var_p_flat, dim=1)
        kl = log_terms + trace_terms + quad
        return kl

# ----------------------------
# ELBO computation
# ----------------------------

def compute_ddpm_elbo_batch(x0: torch.Tensor,
                            x_t: torch.Tensor,
                            # x_t_pred: torch.Tensor,
                            t: int,
                            betas: torch.Tensor,
                            model_output: Dict[str, torch.Tensor],
                            mode: str = "eps",   # "eps" or "mu"
                            sigma_model: Optional[torch.Tensor] = None,
                            sigma_special_default: str = "tilde_beta"):
    """
    Compute exact ELBO components for a minibatch at a single timestep t.
    - x0: (B, *data_shape) ground truth
    - x_t: (B, *data_shape) noisy sample at time t drawn from q(x_t | x0)
    - t: integer timestep (0-based index; corresponds to beta_{t+1} in paper)
    - betas: (T,) schedule of betas
    - model_output: if mode=="eps": {'eps': eps_theta} where eps_theta shape = x_t shape
                    if mode=="mu": {'mu': mu_theta, 'sigma': sigma_theta} (sigma optional)
    - sigma_model: optional tensor (T,) specifying model variance schedule; if None and mode=="eps"
                   we default to sigma = tilde_beta_t (matching q posterior var)
    - sigma_special_default: "tilde_beta" means use tilde_beta_t for p variance if not provided.
    Returns dict with:
      L0: reconstruction term (if we can compute; else None)
      L_t: KL(q(x_{t-1}|x_t,x0) || p_theta(x_{t-1}|x_t)) per-sample (B,)
      L_T: prior KL (scalar per-sample) when t == T-1 we return it computed; else computed separately
      elbo: total negative ELBO per-sample (approx for the batch -- this function handles single t)
    Note: This function computes the KL term for the given t (1..T-1). To get full ELBO sum over t,
    sum over sampled timesteps or loop over t.
    """
    device = x0.device
    betas = betas.to(t.device)
    T = betas.shape[0]
    if not (0 <= t < T):
        raise ValueError("t must be in [0, T-1] (0-based indexing).")

    alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod = compute_alphas(betas)

    alphas = alphas.to(t.device)
    alphas_cumprod = alphas_cumprod.to(t.device)
    # alpha_t = alphas[t]
    # bar_alpha_t = alphas_cumprod[t]

    # compute true posterior params q(x_{t-1}|x_t,x0)
    # note: q_posterior_mean_variance expects t>=1 for proper tilde_beta formula; for t=0 it is unused.
    if t == 0:
        raise ValueError("KL for t=0 is not defined here (t must be >=1). This function computes L_t for t>=1.")

    # true posterior mean and variance
    # print('inside')
    # print(torch.mean(x0))
    # print(torch.mean(x_t))
    # print(t)
    # print('-----')
    mu_q, tilde_beta_t = q_posterior_mean_variance(x_t, x0, t, betas, alphas, alphas_cumprod)
    var_q = tilde_beta_t  # scalar

    # Model's p_theta(x_{t-1} | x_t) parameters:
    if mode == "mu":
        mu_p = model_output['mu']  # (B, *data_shape)
        sigma_p = model_output.get('sigma', None)
        if sigma_p is None:
            # fallback: use provided sigma_model schedule at t, or tilde_beta_t
            if sigma_model is not None:
                sigma_p_t = float(sigma_model[t].item())
            else:
                sigma_p_t = tilde_beta_t if sigma_special_default == "tilde_beta" else float(1e-2)
            var_p = sigma_p_t
        else:
            var_p = sigma_p  # allow scalar or broadcastable tensor
    elif mode == "eps":
        # model gives predicted noise eps_theta; convert to x0_pred then set mu_p = \tilde\mu_t(x_t, x0_pred)
        eps_theta = model_output  # (B, *data_shape)
        # compute x0_pred = (x_t - sqrt(1 - bar_alpha_t) * eps_theta) / sqrt(bar_alpha_t)
        # guard numerical stability:
        sqrt_bar_alpha_t = sqrt_alphas_cumprod[t]
        x0_pred = (x_t - sqrt_1m_alphas_cumprod[t].view(1, *([1] * (x_t.dim()-1))) * eps_theta) / sqrt_bar_alpha_t.view(1, *([1] * (x_t.dim()-1)))
        
        mu_p, _ = q_posterior_mean_variance(x_t, x0_pred, t, betas, alphas, alphas_cumprod)
        # variance choice for model: try sigma_model if given else default to tilde_beta_t
        if sigma_model is not None:
            var_p = float(sigma_model[t].item())
        else:
            var_p = tilde_beta_t if sigma_special_default == "tilde_beta" else float(1e-2)
    else:
        raise ValueError("Unsupported mode. Choose 'eps' or 'mu'.")

    # compute KL for this t per sample
    L_t = gaussian_kl_diag(mu_q, var_q, mu_p, var_p)  # (B,)

    # L_0: reconstruction term: -E_q[log p_theta(x0 | x1)]
    # Can compute if model provides a p_theta(x0|x1) mean and variance (i.e., mode=='mu' and t==1)
    # But commonly L_0 uses p(x0|x1). We'll not compute it here except when model provides mu for t=1.
    L_0 = None
    if mode == "mu" and ('recon_mu' in model_output):
        # recon_mu is model's p_theta(x0 | x1) mean, sigma0 provided optionally in model_output
        recon_mu = model_output['recon_mu']
        sigma0 = model_output.get('sigma0', 1.0)
        # negative log-likelihood under Gaussian p(x0|x1)
        B = x0.shape[0]
        d = x0[0].numel()
        recon_term = 0.5 * (d * torch.log(2 * torch.pi * sigma0) + torch.sum((x0 - recon_mu).view(B, -1) ** 2, dim=1) / sigma0)
        L_0 = recon_term  # (B,)

    # L_T: KL(q(x_T | x0) || p(x_T)) where p(x_T) = N(0, I)
    # q(x_T|x0) = N(sqrt(bar_alpha_T) x0, (1 - bar_alpha_T) I)
    bar_alpha_T = alphas_cumprod[-1]
    var_qT = float((1.0 - bar_alpha_T).item())
    # KL to N(0,I): 0.5 ( -d + tr(var_qT * I) + ||sqrt(bar_alpha_T) x0||^2 + d*log(1) - d*log(var_qT) ) simplified below
    B = x0.shape[0]
    d = x0[0].numel()
    x0_flat_sq = torch.sum((torch.sqrt(bar_alpha_T).view(1, *([1] * (x0.dim()-1))) * x0).view(B, -1) ** 2, dim=1)
    L_T = 0.5 * (-d + d * var_qT + x0_flat_sq + d * (-torch.log(torch.tensor(var_qT, device=x0.device))))
    # note: L_T is (B,)

    # total negative ELBO for this particular timestep's KL plus (optionally) reconstruction and prior terms:
    # Real ELBO is sum over all t of L_t plus L_0 and L_T; here return values so user can sum across sampled t's.
    return {
        'L0': L_0,         # (B,) or None
        'L_t': L_t,        # (B,)
        'L_T': L_T,        # (B,)
        'mu_q': mu_q,
        'var_q': var_q,
        'mu_p': mu_p,
        'var_p': var_p,
    }

# ----------------------------
# Minimal runnable example
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cpu")

    # toy parameters
    T = 10
    beta_start, beta_end = 1e-4, 0.02
    betas = make_beta_schedule(beta_start, beta_end, T, device=device)  # (T,)
    alphas, alphas_cumprod, _, _ = compute_alphas(betas)

    B = 4
    C, H, W = 3, 16, 16
    x0 = torch.randn(B, C, H, W, device=device)

    # sample a timestep t (1..T-1)
    t = 4  # 0-based index; corresponds to paper's t+1
    sqrt_bar_alpha_t = torch.sqrt(alphas_cumprod[t])
    sqrt_1m_bar_alpha_t = torch.sqrt(1.0 - alphas_cumprod[t])

    # sample x_t from q(x_t | x0)
    eps = torch.randn_like(x0)
    x_t = sqrt_bar_alpha_t.view(1,1,1,1) * x0 + sqrt_1m_bar_alpha_t.view(1,1,1,1) * eps

    # pretend our model predicts eps_theta
    # For demonstration, set eps_theta slightly different from ground truth noise
    eps_theta = eps + 0.1 * torch.randn_like(eps)

    model_output = {'eps': eps_theta}  # mode='eps'

    out = compute_ddpm_elbo_batch(x0, x_t, t, betas, model_output, mode="eps")
    print("L_t per sample:", out['L_t'])
    print("mean L_t:", out['L_t'].mean().item())
    print("L_T per sample:", out['L_T'])
    # L0 is None in this mode unless recon_mu provided

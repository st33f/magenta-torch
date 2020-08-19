import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ELBO(pred, target, mu, sigma, free_bits):
    """
    Evidence Lower Bound
    Return KL Divergence and KL Regularization using free bits
    """
    device = pred.device
    # Reconstruction error
    # Pytorch cross_entropy combines LogSoftmax and NLLLoss
    likelihood = -binary_cross_entropy(pred, target, reduction='sum')
    print()
    print(f"likelihood: {likelihood}")
    print(f"bce: {binary_cross_entropy(pred, target, reduction='sum')}")
    # Regularization error
    sigma_prior = torch.tensor([1], dtype=torch.float, device=device)
    mu_prior = torch.tensor([0], dtype=torch.float, device=device)
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)
    #print(f"kl div: {kl_div}")
    print(f"mean KL div: {torch.mean(kl_div)}")
    elbo = torch.mean(likelihood) - torch.max(torch.mean(kl_div)-free_bits, torch.tensor([0], dtype=torch.float, device=device))
    print(f"elbo: {elbo}")
    print(f"-elbo: {-elbo}")

    return -elbo, kl_div.mean()


def custom_ELBO(pred, target, mu, sigma, free_bits):
    """
    Attempting to fix this loss function
    """
    device = pred.device
    r_loss = binary_cross_entropy_with_logits(pred, target, reduction='sum')
    # Regularization error
    sigma_prior = torch.tensor([1], dtype=torch.float, device=device)
    mu_prior = torch.tensor([0], dtype=torch.float, device=device)
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)

    kl_cost = torch.max(torch.mean(kl_div) - free_bits, torch.tensor([0], dtype=torch.float, device=device))
    return r_loss.to(device), kl_cost.to(device), kl_div.to(device)
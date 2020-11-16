import torch
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy, mse_loss
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hamming_distance(s1, s2) -> int:
    """Return the Hamming distance between equal-length sequences."""
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length.")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

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
    #r_loss = binary_cross_entropy_with_logits(pred, target, reduction='sum')
    r_loss = binary_cross_entropy(pred, target, reduction='sum')
    r_loss = r_loss.to(device)
    #device = pred.device
    #pred = pred.to(device)
    #target = target.to(device)
    print(f"target: {target.size()}")
    print(f"pred: {pred.size()}")
    # Regularization error
    sigma_prior = torch.tensor([1], dtype=torch.float, device=device)
    mu_prior = torch.tensor([0], dtype=torch.float, device=device)
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)
    kl_cost = torch.max(torch.mean(kl_div) - free_bits, torch.tensor([0], dtype=torch.float, device=device))

    # create flat prediction ( one hot reconstruction ) and additional metrics
    with torch.no_grad():
        pred_max = torch.argmax(pred, dim=2)
        flat_pred = torch.zeros(pred.size(), device=device)
        batch_size = list(pred.size())[1]
        for i in range(256):
            for j in range(batch_size):
                # print(argmax[i])
                flat_pred[i, j, pred_max[i, j]] = 1

        # calc hamming distance
        hamming_dist = hamming_distance(target.argmax(-1), flat_pred.argmax(-1))
        norm_ham_dist = float(hamming_dist.sum() / len(hamming_dist))

        # calculate accuracy per batch, over all timesteps as classifier accuracy
        target = target.cpu()
        flat_pred = flat_pred.cpu()
        acc = (target.argmax(-1) == flat_pred.argmax(-1)).float().detach().numpy()
        acc_percentage = float(100 * acc.sum() / (len(acc) * batch_size))
        #print(f"Accuracy: {acc_percentage} %")

    return r_loss.to(device), kl_cost.to(device), kl_div.to(device), \
           norm_ham_dist, acc_percentage



def flat_ELBO(pred, target, mu, sigma, free_bits):
    #pred = torch.transpose(pred, 0, 1)
    #target = torch.transpose(target, 0, 1)

    # create flat prediction - get indexes
    flat_pred = torch.argmax(pred, dim=2)
    flat_target = torch.argmax(target, dim=2)

    print("FLAT TARGETS")
    print(flat_pred.shape)
    print(flat_pred)
    print()
    print(flat_target.shape)
    print(flat_target)

    # pass to Cross entropy, divide by batch size
    print(pred.shape)
    #r_loss = cross_entropy(pred, flat_target, reduction='sum')
    r_loss = mse_loss(flat_pred.float(), flat_target.float(), reduction="sum")
    # Regularization error
    sigma_prior = torch.tensor([1], dtype=torch.float, device=device)
    mu_prior = torch.tensor([0], dtype=torch.float, device=device)
    p = Normal(mu_prior, sigma_prior)
    q = Normal(mu, sigma)
    kl_div = kl_divergence(q, p)
    kl_cost = torch.max(torch.mean(kl_div) - free_bits, torch.tensor([0], dtype=torch.float, device=device))

    return r_loss.to(device), kl_cost.to(device), kl_div.to(device)

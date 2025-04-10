"""
Modified by https://github.com/GAIR-NLP/MAYE/tree/master/maye/rlhf
For more details, visit: https://arxiv.org/abs/2504.02587
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PPOLoss(nn.Module):
    """
    PPO Loss for RLHF fine-tuning
    """
    def __init__(
        self,
        epsilon_low=0.2,
        epsilon_high=0.2,
        kl_loss_coeff=0.01,
    ):
        super().__init__()
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.kl_loss_coeff = kl_loss_coeff

    def forward(
        self,
        values: torch.Tensor,
        old_logprobs: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        kl_div: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        # Calculate importance ratio
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Apply mask if provided
        if mask is not None:
            ratio = ratio * mask
            advantages = advantages * mask
        
        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1 - self.epsilon_low, 1 + self.epsilon_high
        ) * advantages

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        if values is not None and mask is not None:
            value_loss = F.mse_loss(values, advantages + old_logprobs, reduction='none')
            value_loss = (value_loss * mask).mean()
        else:
            value_loss = torch.tensor(0.0, device=policy_loss.device)
        
        # KL divergence loss
        if kl_div is not None and self.kl_loss_coeff > 0:
            kl_loss = self.kl_loss_coeff * kl_div.mean()
        else:
            kl_loss = torch.tensor(0.0, device=policy_loss.device)
        
        # Total loss
        total_loss = policy_loss + value_loss + kl_loss
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_loss": kl_loss,
        }

def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    """
    Compute generalized advantage estimation (GAE)
    """
    last_gae_lam = 0
    advantages = torch.zeros_like(rewards)
    
    # Reverse iteration for GAE calculation
    for t in reversed(range(len(rewards))):
        # For the last step, next_value is 0
        next_value = values[t + 1] if t < len(values) - 1 else 0
        
        # Calculate delta (TD error)
        delta = rewards[t] + gamma * next_value * mask[t] - values[t]
        
        # Calculate advantage with GAE
        last_gae_lam = delta + gamma * lam * mask[t] * last_gae_lam
        advantages[t] = last_gae_lam
    
    return advantages

def compute_rewards(
    responses: List[str],
    references: List[str],
    device,
    whiten_rewards: bool = False,
    min_response_length: int = 18,
    penalise_no_eos: bool = True,
    reward_penalty: float = -0.1,
):
    """
    Compute rewards for generated responses
    """
    device = torch.device("cuda:0")

    # Simple reward computation based on response length for demonstration
    # In a real implementation, this would use a trained reward model
    rewards = []
    
    for response, reference in zip(responses, references):
        # Basic length check
        if len(response) < min_response_length:
            reward = reward_penalty
        else:
            # Simple reward based on overlap with reference
            # This is just for demonstration - real reward models are more complex
            reward = len(set(response.split()) & set(reference.split())) / max(len(reference.split()), 1)
        
        rewards.append(reward)
    
    # Convert to tensor
    rewards = torch.tensor(rewards, device=device)
    
    # Optionally normalize rewards
    if whiten_rewards and len(rewards) > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    return rewards


def calculate_kl_divergence(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    Calculate KL divergence between policy and reference policy
    """
    kl_div = logprobs - ref_logprobs
    
    if mask is not None:
        kl_div = kl_div * mask
    
    return kl_div
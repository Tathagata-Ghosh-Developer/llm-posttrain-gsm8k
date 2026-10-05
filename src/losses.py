"""The three training objectives, written out by hand."""
import torch
import torch.nn.functional as F


def sft_loss(logits, labels):
    """Shifted next-token cross-entropy, ignoring positions labelled -100.

    logits: (B, T, V); the logit at position t predicts token t+1.
    labels: (B, T) token ids, with -100 on prompt and padding positions.
    Returns (sum of token NLLs, number of target tokens) so that the caller
    can normalise over a whole gradient-accumulated batch.
    """
    logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = torch.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (nll * mask).sum(), mask.sum()


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """DPO loss from summed response log-probs, each of shape (B,).

    Implicit reward r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x)).
    loss = -log sigmoid(r(x, y_chosen) - r(x, y_rejected)), averaged over pairs.
    """
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward.detach(), rejected_reward.detach()


def group_advantages(rewards, eps=1e-4):
    """GRPO advantage: reward standardised within each group of G samples.

    rewards: (num_questions, G). A group where every sample gets the same
    reward has zero advantage, so it carries no policy-gradient signal.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps, kl_beta):
    """Clipped surrogate plus KL penalty, per token, averaged within each sequence.

    logp, old_logp, ref_logp, mask: (B, T) over completion tokens.
    advantages: (B,), one value per sampled completion.
    Returns the per-sequence loss (B,) and a dict of diagnostics.
    """
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    # k3 estimator of KL(pi || pi_ref): exp(q - p) - (q - p) - 1 >= 0, unbiased under samples of pi
    log_ref_ratio = ref_logp - logp
    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    per_token = -(surrogate - kl_beta * kl)
    n_tokens = mask.sum(dim=1).clamp(min=1)
    per_sequence = (per_token * mask).sum(dim=1) / n_tokens
    clipped = ((ratio - 1).abs() > clip_eps).float()
    stats = {
        "kl": ((kl * mask).sum() / mask.sum()).item(),
        "clip_frac": ((clipped * mask).sum() / mask.sum()).item(),
    }
    return per_sequence, stats

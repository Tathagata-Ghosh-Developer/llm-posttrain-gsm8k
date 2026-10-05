import math

import torch

from src.common import build_batch
from src.losses import dpo_loss, group_advantages, grpo_loss, sft_loss


def test_sft_loss_uniform_logits_is_log_vocab():
    vocab = 11
    logits = torch.zeros(2, 5, vocab)
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 1, 2, -100, -100]])
    total, n = sft_loss(logits, labels)
    assert n.item() == 5  # targets are labels[:, 1:] that are not -100
    assert math.isclose(total.item() / n.item(), math.log(vocab), rel_tol=1e-6)


def test_sft_loss_ignores_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 7)
    labels = torch.tensor([[-100, -100, -100, 2, 5, -100]])
    base, _ = sft_loss(logits, labels)
    changed = logits.clone()
    # log_softmax is shift-invariant, so perturb a single vocabulary entry
    changed[0, 0, 2] += 10.0  # position 0 predicts token 1, a prompt token
    changed[0, 5, 2] += 10.0  # the last position predicts nothing
    other, _ = sft_loss(changed, labels)
    assert torch.allclose(base, other)
    changed[0, 3, 0] += 10.0  # position 3 predicts token 4, a target: loss must change
    assert not torch.allclose(base, sft_loss(changed, labels)[0])


def test_sft_loss_uses_shifted_targets():
    logits = torch.full((1, 3, 4), -1e4)
    logits[0, 0, 2] = 0.0  # position 0 is certain of token 2
    logits[0, 1, 3] = 0.0  # position 1 is certain of token 3
    labels = torch.tensor([[-100, 2, 3]])
    total, n = sft_loss(logits, labels)
    assert n.item() == 2 and total.item() < 1e-6


def test_dpo_loss_is_log2_when_policy_equals_reference():
    lp = torch.tensor([-10.0, -20.0])
    loss, rc, rr = dpo_loss(lp, lp - 3, lp, lp - 3, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert torch.all(rc == 0) and torch.all(rr == 0)


def test_dpo_loss_sign():
    ref_c, ref_r = torch.tensor([-10.0]), torch.tensor([-10.0])
    better, _, _ = dpo_loss(torch.tensor([-9.0]), torch.tensor([-11.0]), ref_c, ref_r, beta=0.1)
    worse, _, _ = dpo_loss(torch.tensor([-11.0]), torch.tensor([-9.0]), ref_c, ref_r, beta=0.1)
    assert better.item() < math.log(2) < worse.item()
    # gradient raises the chosen log-prob and lowers the rejected one
    pc = torch.tensor([-10.0], requires_grad=True)
    pr = torch.tensor([-10.0], requires_grad=True)
    dpo_loss(pc, pr, ref_c, ref_r, beta=0.1)[0].backward()
    assert pc.grad.item() < 0 < pr.grad.item()


def test_group_advantages():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.all(adv[1] == 0)  # identical rewards -> no signal
    assert adv[0, 0] > 0 > adv[0, 1]


def test_grpo_loss_on_policy_gradient_direction():
    logp = torch.tensor([[-1.0, -2.0], [-1.0, -2.0]], requires_grad=True)
    mask = torch.ones(2, 2)
    adv = torch.tensor([1.0, -1.0])
    per_seq, stats = grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, clip_eps=0.2, kl_beta=0.1)
    assert torch.allclose(per_seq, -adv)  # ratio = 1 and KL = 0 at the sampling policy
    assert stats["kl"] == 0.0
    per_seq.sum().backward()
    assert torch.all(logp.grad[0] < 0) and torch.all(logp.grad[1] > 0)  # raise good, lower bad


def test_grpo_loss_clipping_stops_gradient():
    old = torch.zeros(1, 1)
    logp = torch.tensor([[0.5]], requires_grad=True)  # ratio e^0.5 > 1.2
    per_seq, stats = grpo_loss(logp, old, logp.detach(), torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.0)
    per_seq.sum().backward()
    assert logp.grad.abs().item() == 0.0 and stats["clip_frac"] == 1.0


def test_grpo_kl_penalty_is_positive_off_reference():
    logp = torch.tensor([[-1.0, -1.0]])
    per_seq, stats = grpo_loss(logp, logp, logp - 0.5, torch.tensor([0.0]), torch.ones(1, 2), 0.2, 1.0)
    assert stats["kl"] > 0 and per_seq.item() > 0


def test_build_batch_masks_only_completion_tokens():
    ids, attn, mask = build_batch([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], pad_id=0, device="cpu")
    assert ids.tolist() == [[1, 2, 3, 5, 6], [4, 7, 8, 9, 0]]
    assert attn.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    # entry t says whether token t+1 is a completion token
    assert mask.tolist() == [[0, 0, 1, 1], [1, 1, 1, 0]]

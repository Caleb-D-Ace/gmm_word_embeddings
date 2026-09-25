import math

import torch

from gmm_word_embedding import GMMWordEmbedding


def test_get_word_params_shapes_and_constraints():
    model = GMMWordEmbedding(vocab_size=10, embedding_dim=4, K=3)
    word_ids = torch.tensor([0, 1, 2])

    mu, var, mix_weights = model.get_word_params(word_ids)

    assert mu.shape == (3, 3, 4)
    assert var.shape == (3, 3, 4)
    assert torch.all(var > 0)
    assert mix_weights.shape == (3, 3)
    assert torch.allclose(mix_weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_log_overlap_matches_closed_form_for_identical_unit_gaussians():
    # D=1, mu1=mu2=0, var1=var2=1 has a hand-computable closed form:
    # -0.5 * log(var1+var2) - 0.5*log(2*pi) == -0.5 * log(4*pi)
    mu = torch.zeros(1, 1, 1)
    var = torch.ones(1, 1, 1)

    overlap = GMMWordEmbedding.log_overlap(mu, mu, var, var)

    expected = -0.5 * math.log(4 * math.pi)
    assert torch.allclose(overlap, torch.tensor([[[expected]]]), atol=1e-5)


def test_gmm_energy_reduces_to_log_overlap_for_single_component():
    # With K=1 and mixture weights of 1, log(exp(x) * 1 * 1) == x
    log_overlap_matrix = torch.tensor([[[-1.5]]])
    p1 = torch.ones(1, 1)
    p2 = torch.ones(1, 1)

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    assert torch.allclose(energy, torch.tensor([-1.5]), atol=1e-5)


def test_gmm_energy_is_exact_for_very_negative_log_overlaps():
    # At D=50 real log-overlaps sit around -60. The energy must track them rather than
    # bottoming out at a fixed floor (the old exp -> sum + 1e-8 -> log version returned -18.42 here).
    log_overlap_matrix = torch.tensor([[[-60.0]]])
    p1 = torch.ones(1, 1)
    p2 = torch.ones(1, 1)

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    assert torch.allclose(energy, torch.tensor([-60.0]), atol=1e-4)


def test_gmm_energy_matches_direct_computation_with_multiple_components():
    log_overlap_matrix = torch.tensor([[[-2.0, -3.0], [-4.0, -5.0]]])
    p1 = torch.tensor([[0.7, 0.3]])
    p2 = torch.tensor([[0.4, 0.6]])

    energy = GMMWordEmbedding.gmm_energy(log_overlap_matrix, p1, p2)

    weights = torch.tensor([[0.7 * 0.4, 0.7 * 0.6], [0.3 * 0.4, 0.3 * 0.6]], dtype=torch.float64)
    expected = torch.log((weights * torch.exp(log_overlap_matrix[0].double())).sum())
    assert torch.allclose(energy.double(), expected.unsqueeze(0), atol=1e-5)


def test_high_dimensional_model_still_receives_gradients():
    # Regression: at the default embedding_dim=50 the energies used to be pinned at an
    # epsilon floor, giving gradients around 1e-19, so training silently never learned.
    torch.manual_seed(0)
    model = GMMWordEmbedding(vocab_size=20, embedding_dim=50, K=2)

    energy = model.forward(torch.arange(10), torch.arange(10, 20))
    energy.sum().backward()

    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 1e-3


def test_forward_output_shape_and_gradients_flow():
    model = GMMWordEmbedding(vocab_size=10, embedding_dim=4, K=2)
    target_ids = torch.tensor([0, 1])
    ctx_ids = torch.tensor([2, 3])

    energy = model.forward(target_ids, ctx_ids)
    assert energy.shape == (2,)

    energy.sum().backward()
    assert model.mu_embeddings.weight.grad is not None
    touched_rows = model.mu_embeddings.weight.grad[[0, 1, 2, 3]]
    assert torch.any(touched_rows != 0)


def test_max_margin_ranking_zero_when_margin_already_satisfied():
    E_pos = torch.tensor([5.0, 5.0])
    E_neg = torch.tensor([0.0, 0.0])

    loss = GMMWordEmbedding.max_margin_ranking(E_pos, E_neg, margin=1.0)

    assert loss.item() == 0.0


def test_max_margin_ranking_equals_margin_when_energies_tied():
    E_pos = torch.zeros(3)
    E_neg = torch.zeros(3)

    loss = GMMWordEmbedding.max_margin_ranking(E_pos, E_neg, margin=2.0)

    assert math.isclose(loss.item(), 2.0)

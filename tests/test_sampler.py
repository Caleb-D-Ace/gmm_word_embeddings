import torch

from sampler import NegativeSampler


def test_weights_sum_to_one():
    sampler = NegativeSampler(word_counts=[1, 2, 3, 4], device=torch.device("cpu"))

    assert torch.allclose(sampler.weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_weights_increase_with_frequency():
    sampler = NegativeSampler(word_counts=[1, 2, 3, 4], device=torch.device("cpu"))

    weights = sampler.weights.tolist()
    assert weights == sorted(weights)  # counts are increasing, so weights should be too


def test_sample_returns_valid_ids_in_range():
    sampler = NegativeSampler(word_counts=[1, 2, 3, 4], device=torch.device("cpu"))

    samples = sampler.sample(100)

    assert samples.shape == (100,)
    assert samples.dtype == torch.int64
    assert torch.all(samples >= 0)
    assert torch.all(samples < 4)


def test_default_device_resolves_to_cpu_or_cuda():
    sampler = NegativeSampler(word_counts=[1, 1, 1])

    assert sampler.device.type in ("cpu", "cuda")

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SkipGramDataset


def _make_bin_file(path, values, dtype=np.uint16):
    np.array(values, dtype=dtype).tofile(path)
    return path


def test_len_accounts_for_window_on_both_sides(tmp_path):
    bin_path = _make_bin_file(tmp_path / "corpus.bin", list(range(10)))
    dataset = SkipGramDataset(bin_file=str(bin_path), window_size=2)

    assert len(dataset) == 10 - 2 * 2


def test_getitem_returns_correct_center_and_context(tmp_path):
    bin_path = _make_bin_file(tmp_path / "corpus.bin", list(range(10)))
    dataset = SkipGramDataset(bin_file=str(bin_path), window_size=2)

    center, ctx = dataset[0]

    assert center.item() == 2  # data[window_size]
    assert ctx.tolist() == [0, 1, 3, 4]  # left window + right window
    assert center.dtype == torch.long
    assert ctx.dtype == torch.long


def test_dataloader_batches_have_expected_shapes(tmp_path):
    bin_path = _make_bin_file(tmp_path / "corpus.bin", list(range(20)))
    dataset = SkipGramDataset(bin_file=str(bin_path), window_size=3)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    centers, contexts = next(iter(loader))

    assert centers.shape == (4,)
    assert contexts.shape == (4, 2 * 3)

import pytest
import torch

from gmm_training import GmmTrainer, TrainingConfig


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.processed_dir == "data/processed"
    assert config.output_path == "data/model"  # directory: gmm_embeddings.npz and .pt are both written here
    assert config.embedding_dim == 50
    assert config.K == 2
    assert config.window_size == 5
    assert config.batch_size == 256
    assert config.epochs == 5
    assert config.optimizer == "adam"
    assert config.lr == 0.01
    assert config.lr_final == 1e-5
    assert config.margin == 1.0
    assert config.num_negatives == 1  # pinned until multi-negative training is implemented


@pytest.mark.parametrize("name, expected", [("adagrad", torch.optim.Adagrad), ("adam", torch.optim.Adam)])
def test_make_optimizer_builds_the_requested_optimizer(name, expected):
    trainer = GmmTrainer(TrainingConfig(optimizer=name))
    params = list(torch.nn.Linear(2, 2).parameters())

    assert isinstance(trainer.make_optimizer(params), expected)


def test_make_optimizer_rejects_unknown_names():
    trainer = GmmTrainer(TrainingConfig(optimizer="sgd"))

    with pytest.raises(ValueError):
        trainer.make_optimizer(list(torch.nn.Linear(2, 2).parameters()))


def test_trainer_init_stores_config_and_resolves_device():
    config = TrainingConfig()
    trainer = GmmTrainer(config)

    assert trainer.config is config
    assert isinstance(trainer.device, torch.device)
    assert trainer.device.type in ("cpu", "cuda")


def test_train_raises_when_processed_dir_missing(tmp_path):
    config = TrainingConfig(processed_dir=str(tmp_path / "does_not_exist"))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_train_raises_when_vocab_file_missing(tmp_path):
    (tmp_path / "corpus_index.bin").touch()
    config = TrainingConfig(processed_dir=str(tmp_path))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_train_raises_when_corpus_bin_missing(tmp_path):
    (tmp_path / "sorted_vocab.json").write_text("{}", encoding="utf-8")
    config = TrainingConfig(processed_dir=str(tmp_path))
    trainer = GmmTrainer(config)

    with pytest.raises(FileNotFoundError):
        trainer.train()


def test_reshape_pairs_each_center_with_its_flattened_context():
    # Spec agreed on separately: repeat_interleave the centers so each one
    # lines up with every word in its own context window after flattening.
    centers = torch.tensor([5, 9])
    contexts = torch.tensor([
        [1, 2, 3, 4],
        [6, 7, 8, 9],
    ])

    target_ids, ctx_ids = GmmTrainer.reshape(centers, contexts)

    assert torch.equal(target_ids, torch.tensor([5, 5, 5, 5, 9, 9, 9, 9]))
    assert torch.equal(ctx_ids, torch.tensor([1, 2, 3, 4, 6, 7, 8, 9]))

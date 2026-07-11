from __future__ import annotations

from pathlib import Path


import pytest
import torch
import yaml


ABLATION_NAMES = [
    "100m_modulo_balanced_secondary_shared",
    "100m_modulo_balanced_secondary_no_shared",
    "100m_modulo_shared",
    "100m_random_shared",
    "100m_round_robin_shared",
    "100m_shared_only",
    "100m_dense_residual",
]


def test_token_routed_supports_explicit_lexical_routing_strategies():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    freqs = torch.tensor([100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0])

    zipf = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="zipf",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    modulo = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="modulo",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    random_a = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    random_b = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()
    round_robin = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="round_robin",
            token_frequencies=freqs,
            shared_expert=False,
        )
    ).token_to_expert.cpu()

    assert not torch.equal(zipf, modulo)
    assert torch.equal(random_a, random_b)
    assert not torch.equal(random_a, modulo)
    assert sorted(round_robin.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_topk_auxiliary_routes_preserve_control_strategy():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    freqs = torch.tensor([100.0, 90.0, 80.0, 70.0, 4.0, 3.0, 2.0, 1.0])

    modulo = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="modulo",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()
    random_a = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()
    random_b = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=8,
            routing_strategy="random",
            token_frequencies=freqs,
            top_k=2,
            shared_expert=False,
        )
    ).topk_token_to_expert.cpu()

    assert torch.equal(modulo[1], (modulo[0] + 1) % 4)
    assert torch.equal(random_a, random_b)
    assert torch.all(random_a[0] != random_a[1])


def test_zipf_without_frequencies_fails_instead_of_silent_modulo_fallback():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    with pytest.raises(ValueError, match="requires token_frequencies"):
        TokenRoutedMLP(
            MLPConfig(
                hidden_size=8,
                intermediate_size=16,
                num_experts=4,
                vocab_size=32,
                routing_strategy="zipf",
                token_frequencies=None,
                top_k=2,
                shared_expert=False,
            )
        )


def test_modulo_balanced_secondary_reproduces_fixed_top2_lookup_without_router():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    mlp = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=32,
            routing_strategy="modulo_balanced_secondary",
            token_frequencies=None,
            top_k=2,
            shared_expert=False,
        )
    )
    routes = mlp.topk_token_to_expert.cpu()

    assert torch.all(routes[0] != routes[1])
    assert torch.bincount(routes[0], minlength=4).tolist() == [8, 8, 8, 8]
    assert torch.bincount(routes[1], minlength=4).tolist() == [8, 8, 8, 8]


def test_model_config_and_o200k_parser_support_ablation_switches():
    from complexity.config import ModelConfig
    from complexity.training.o200k_pretrain import build_parser, make_config

    args = build_parser().parse_args([
        "--routing-strategy", "random",
        "--no-shared-expert",
    ])
    args.vocab_size = 200019
    profile = {
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "shared_intermediate_size": 1536,
    }
    for key, value in profile.items():
        setattr(args, key, value)

    config = make_config(args)

    assert ModelConfig(routing_strategy="random").routing_strategy == "random"
    assert config.routing_strategy == "random"
    assert config.shared_expert is False


def test_seven_100m_ablation_yaml_configs_match_reported_two_gpu_runs():
    from complexity.training.o200k import build_parser
    from complexity.training.run_config import parse_args_with_yaml_config

    root = Path("configs/run_configs/ablations_100m")
    expected = {f"{name}.yaml" for name in ABLATION_NAMES}
    expected_strategy = {
        "100m_modulo_balanced_secondary_shared": "modulo_balanced_secondary",
        "100m_modulo_balanced_secondary_no_shared": "modulo_balanced_secondary",
        "100m_modulo_shared": "modulo",
        "100m_random_shared": "random",
        "100m_round_robin_shared": "round_robin",
        "100m_shared_only": "modulo_balanced_secondary",
        "100m_dense_residual": "modulo_balanced_secondary",
    }

    found = {p.name for p in root.glob("*.yaml")}

    assert expected <= found
    for name in ABLATION_NAMES:
        path = root / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())["run"]
        args = parse_args_with_yaml_config(build_parser(), ["--config", str(path)])
        assert data["profile"] == "100m"
        assert data["dataset"] == "fineweb"
        assert data["steps"] == 954
        assert data["batch_size"] == 256
        assert data["seq_len"] == 2048
        assert data["run_name"] == f"b200-1b-{name}"
        assert args.routing_strategy == expected_strategy[name]
        assert args.steps == 954
        assert args.batch_size == 256
        assert args.seq_len == 2048
        assert args.top_k == (1 if name in {"100m_shared_only", "100m_dense_residual"} else 2)
        if name not in {"100m_shared_only", "100m_dense_residual"}:
            assert args.top_k_primary_weight == 0.5
        assert data["save_dir"].endswith(data["run_name"])


def test_300m_entrypoint_defaults_match_verified_checkpoint(monkeypatch):
    import scripts.train_300m_tr_local as train_300m

    captured = {}
    original_parse_args = train_300m.argparse.ArgumentParser.parse_args

    def capture_defaults(parser):
        captured["args"] = original_parse_args(parser, [])
        raise RuntimeError("defaults captured")

    monkeypatch.setattr(train_300m.argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(RuntimeError, match="defaults captured"):
        train_300m.main()

    args = captured["args"]
    assert args.tokenizer == "./tokenizer-32k"
    tokenizer = train_300m.Tokenizer.load(args.tokenizer)
    assert tokenizer.vocab_size == 32000
    args.vocab_size = train_300m.infer_vocab_size(args)
    config = train_300m.make_config(args)
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 18
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert config.intermediate_size == 64
    assert config.shared_intermediate_size == 3840
    assert config.num_experts == 4
    assert config.routing_strategy == "modulo_balanced_secondary"
    assert config.top_k == 2
    assert config.top_k_primary_weight == 0.5
    assert config.shared_gate_init == 1.0
    assert config.routed_gate_init == 0.1


def test_seven_100m_ablation_entrypoints_reference_configs():
    root = Path("scripts/ablations_100m")
    expected = {f"train_{name}.sh" for name in ABLATION_NAMES}

    found = {p.name for p in root.glob("train_*.sh")}

    assert expected <= found
    for name in ABLATION_NAMES:
        script = (root / f"train_{name}.sh").read_text()
        assert "scripts/train_100m_o200k_tr_local.py" in script
        assert f"configs/run_configs/ablations_100m/{name}.yaml" in script

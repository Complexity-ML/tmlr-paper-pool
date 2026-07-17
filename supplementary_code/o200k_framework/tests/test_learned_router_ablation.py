from __future__ import annotations

import torch
import torch.nn as nn
import yaml

from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.mixtral_moe import MixtralMoE


def learned_router_config(mode: str) -> MLPConfig:
    return MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        vocab_size=32,
        shared_expert=True,
        shared_intermediate_size=12,
        use_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.5,
        top_k=2,
        router_balance_mode=mode,
        router_aux_loss_coef=0.01,
        router_bias_update_rate=0.1,
    )


def test_learned_router_top2_handles_unbalanced_dispatch():
    mlp = MixtralMoE(learned_router_config("aux_loss"))
    with torch.no_grad():
        mlp.router.weight.zero_()

    hidden = torch.randn(3, 5, 8)
    output = mlp(hidden)

    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    assert mlp.last_topk_expert_ids.shape == (15, 2)
    assert torch.all(mlp.last_topk_expert_ids[:, 0] != mlp.last_topk_expert_ids[:, 1])
    assert int(mlp.last_expert_counts.sum().item()) == 30
    assert int((mlp.last_expert_counts == 0).sum().item()) >= 1


def test_auxiliary_balance_loss_backpropagates_into_router():
    mlp = MixtralMoE(learned_router_config("aux_loss"))
    with torch.no_grad():
        mlp.router.weight.zero_()

    mlp(torch.randn(2, 4, 8))
    aux_loss = mlp.router_auxiliary_loss()
    aux_loss.backward()

    assert aux_loss.requires_grad
    assert mlp.router.weight.grad is not None
    assert mlp.router.weight.grad.abs().sum().item() > 0.0


def test_loss_free_bias_update_favors_underloaded_experts_without_gradients():
    mlp = MixtralMoE(learned_router_config("loss_free_bias"))

    assert not mlp.router_selection_bias.requires_grad
    mlp.update_loss_free_bias(torch.tensor([8.0, 0.0, 0.0, 0.0]))

    assert mlp.router_selection_bias[0] < 0.0
    assert torch.all(mlp.router_selection_bias[1:] > 0.0)
    assert torch.isclose(mlp.router_selection_bias.mean(), torch.tensor(0.0))


def test_auxiliary_and_loss_free_routers_have_identical_trainable_capacity():
    aux = MixtralMoE(learned_router_config("aux_loss"))
    loss_free = MixtralMoE(learned_router_config("loss_free_bias"))

    aux_params = sum(parameter.numel() for parameter in aux.parameters())
    loss_free_params = sum(parameter.numel() for parameter in loss_free.parameters())

    assert aux_params == loss_free_params


def test_o200k_parser_propagates_learned_router_balance_mode():
    from complexity.training.o200k import build_parser, make_config

    args = build_parser().parse_args(
        [
            "--mlp-type",
            "learned_router",
            "--router-balance-mode",
            "loss_free_bias",
            "--router-aux-loss-coef",
            "0.02",
            "--router-bias-update-rate",
            "0.003",
        ]
    )
    args.hidden_size = 8
    args.num_hidden_layers = 2
    args.num_attention_heads = 2
    args.num_key_value_heads = 1
    args.intermediate_size = 16
    args.shared_intermediate_size = 12
    args.vocab_size = 32

    config = make_config(args)

    assert config.mlp_type == "learned_router"
    assert config.router_balance_mode == "loss_free_bias"
    assert config.router_aux_loss_coef == 0.02
    assert config.router_bias_update_rate == 0.003


def test_run_config_counts_gradient_accumulation_in_effective_token_budget():
    from complexity.training.o200k import build_parser
    from complexity.training.run_config import args_to_run_config

    args = build_parser().parse_args(
        [
            "--steps",
            "954",
            "--batch-size",
            "128",
            "--seq-len",
            "2048",
            "--gradient-accumulation-steps",
            "2",
        ]
    )
    run_config = args_to_run_config(
        args,
        model_config={},
        params=1,
        world_size=2,
    )

    assert run_config["tokens_per_step"] == 1_048_576
    assert run_config["total_tokens"] == 1_000_341_504


def test_training_helpers_aggregate_aux_loss_and_update_loss_free_bias():
    from complexity.training.o200k.runtime import (
        learned_router_auxiliary_loss,
        update_loss_free_router_biases,
    )

    aux_router = MixtralMoE(learned_router_config("aux_loss"))
    loss_free_router = MixtralMoE(learned_router_config("loss_free_bias"))
    model = nn.ModuleList([aux_router, loss_free_router])

    aux_router(torch.randn(2, 4, 8))
    aggregated = learned_router_auxiliary_loss(model)

    assert aggregated is not None
    assert aggregated.requires_grad

    loss_free_router.last_expert_counts.copy_(torch.tensor([8.0, 0.0, 0.0, 0.0]))
    updated = update_loss_free_router_biases(model, distributed=False)

    assert updated == 1
    assert loss_free_router.router_selection_bias[0] < 0.0


def test_learned_router_yaml_configs_match_the_100m_budget():
    from complexity.training.o200k import build_parser
    from complexity.training.run_config import parse_args_with_yaml_config

    root = "configs/run_configs/ablations_100m"
    expected = {
        "100m_learned_aux_shared": "aux_loss",
        "100m_learned_loss_free_shared": "loss_free_bias",
    }
    for name, balance_mode in expected.items():
        path = __import__("pathlib").Path(root) / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())["run"]
        args = parse_args_with_yaml_config(build_parser(), ["--config", str(path)])

        assert data["profile"] == "100m"
        assert data["dataset"] == "fineweb"
        assert args.mlp_type == "learned_router"
        assert args.router_balance_mode == balance_mode
        assert args.steps == 954
        assert args.batch_size == 256
        assert args.seq_len == 2048
        assert args.top_k == 2
        assert args.intermediate_size == 768
        assert args.shared_intermediate_size == 768
        assert args.shared_gate_init == 1.0
        assert args.routed_gate_init == 0.5
        assert args.steps * args.batch_size * 2 * args.seq_len == 1_000_341_504


def test_two_gpu_launcher_covers_the_four_central_controls():
    from pathlib import Path

    launcher = Path("scripts/ablations_100m/run_2x_rtx_pro_6000_router_baselines.sh")
    text = launcher.read_text()

    assert "torchrun --standalone --nproc_per_node=2" in text
    assert "100m_modulo_balanced_secondary_shared" in text
    assert "100m_dense_residual" in text
    assert "100m_learned_aux_shared" in text
    assert "100m_learned_loss_free_shared" in text
    assert "--steps 954" in text
    assert "--batch-size 32" in text
    assert "--seq-len 2048" in text
    assert "--gradient-accumulation-steps 8" in text
    assert "--loss-backend liger" in text
    assert "--tokenizer o200k_base" in text
    assert "--vocab-size 200019" in text
    assert "--dataset tokens" in text
    assert "--tokens-path" in text
    assert "tokens.idx.json" in text
    assert "tokens.bin" in text
    assert "rtxpro2-1b-s" in text

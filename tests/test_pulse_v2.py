"""Tests for PULSEv2Model (Hyper-Plastic Resonance Engine): matched 30M parity, strict causality,
O(1) matrix working state, 0-backprop core + Wake-Sleep contrastive anti-hallucination, online W_fast,
and sleep consolidation."""

import torch

from pulse.pulse_v2 import PULSEv2Config, PULSEv2Model
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


class TestPULSEv2ParityAndCausality:
    def test_30m_preset_matches_llama_transformer_parameter_count_closely(self):
        with torch.device("meta"):
            p2 = PULSEv2Model(PULSEv2Config.colab_30m())
            tf = ScalableTransformerLM(ScalableTransformerConfig.colab_30m())

        p_params = p2.num_parameters()
        t_params = tf.num_parameters()
        ratio = p_params / t_params
        assert 0.90 <= ratio <= 1.10, f"PULSE-v2={p_params:,} vs TF={t_params:,} (ratio={ratio:.3f})"
        assert p2.core_param_count() >= 2_500_000

    def test_strict_causality_and_o1_working_state_invariance(self):
        torch.manual_seed(0)
        cfg = PULSEv2Config.micro_test()
        model = PULSEv2Model(cfg)
        model.eval()

        seq_a = torch.randint(4, cfg.vocab_size, (2, 24))
        seq_b = seq_a.clone()
        seq_b[:, 15:] = (seq_b[:, 15:] + 11) % cfg.vocab_size

        with torch.no_grad():
            out_a = model(seq_a)
            out_b = model(seq_b)
            out_long = model(torch.randint(4, cfg.vocab_size, (2, 96)))

        # Strictly causal: future changes at t >= 15 cannot leak to t < 15
        assert torch.allclose(out_a["logits"][:, :15, :], out_b["logits"][:, :15, :], atol=1e-4)

        # Strictly O(1) total working state across 24 vs 96 tokens (DeltaNet + Echoic Ring + SSM)
        assert model.working_state_bytes(out_a["working_state"]) == model.working_state_bytes(
            out_long["working_state"]
        )


class TestPULSEv2LearningPlasticityAndAntiHallucination:
    def test_hybrid_local_training_beats_loss_with_zero_backprop_core_and_high_auroc(self):
        torch.manual_seed(7)
        cfg = PULSEv2Config.micro_test()
        model = PULSEv2Model(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=3e-3)

        # Distinct structured sequences for language modeling + contrastive energy learning
        seqs = torch.stack(
            [
                torch.tensor([5 + i * 10 + (t % 8) for t in range(20)], dtype=torch.long)
                for i in range(6)
            ],
            dim=0,
        )

        losses = []
        for step in range(30):
            st = model.train_step_hybrid_local(
                seqs, io_optimizer=io_opt, async_core=True, core_lr_mult=1.0
            )
            losses.append(st["ce_loss"])

        # STRICT ZERO-BACKPROP CORE AUDIT (field + causal_layer + contrastive_critic)
        assert model.core_autograd_grad_count() == 0
        assert losses[-1] < losses[0] * 0.4

        # Structural anti-hallucination: true continuations vs swapped mismatched continuations
        k = cfg.chunk_size
        prompts = seqs[:, : 16]
        true_cont = seqs[:, 16 : 16 + k]
        false_cont = torch.roll(true_cont, shifts=1, dims=0)
        with torch.no_grad():
            e_true = model.continuation_free_energy(prompts, true_cont, use_fast_weights=False)
            e_false = model.continuation_free_energy(prompts, false_cont, use_fast_weights=False)
            auroc = (e_false.unsqueeze(1) > e_true.unsqueeze(0)).float().mean().item()
        assert auroc >= 0.85

    def test_online_teaching_sleep_consolidation_and_streaming_generation(self):
        torch.manual_seed(17)
        cfg = PULSEv2Config.micro_test()
        cfg.surprise_threshold = 0.05
        cfg.surprise_temperature = 0.02
        cfg.fast_weight_alpha = 0.9
        cfg.fast_weight_decay = 0.0
        cfg.consolidation_lr = 0.14
        model = PULSEv2Model(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=3e-3)

        old_seq = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]], dtype=torch.long)
        for _ in range(15):
            model.train_step_hybrid_local(old_seq, io_optimizer=io_opt)

        new_cue = torch.tensor([[70, 71, 72, 73, 74, 75, 76, 77]], dtype=torch.long)
        new_tgt = torch.tensor([[90, 91, 92, 93]], dtype=torch.long)

        z_truth = model.encode_target_concept(new_tgt)
        err_before = (model.predict_next_concept(new_cue) - z_truth).pow(2).mean().item()
        t_stats = model.teach_fact_online(new_cue, new_tgt, repeats=5)
        err_online = (model.predict_next_concept(new_cue) - z_truth).pow(2).mean().item()

        assert t_stats["max_gate"] > 0.7
        assert err_online < err_before * 0.25

        s_stats = model.sleep_consolidate(num_replay_samples=24, num_epochs=35)
        assert s_stats["causal_consolidated"] >= 1
        assert model.causal_layer.w_fast.norm().item() < 1e-6

        err_sleep = (
            model.predict_next_concept(new_cue, use_fast_weights=False) - z_truth
        ).pow(2).mean().item()
        assert err_sleep < err_before * 0.35

        # Verify O(1) streaming autoregressive generation
        gen = model.generate(new_cue, max_new_tokens=6, temperature=0.0)
        assert gen.shape == (1, 14)

    def test_bf16_autocast_executes_cleanly(self):
        torch.manual_seed(23)
        cfg = PULSEv2Config.micro_test()
        model = PULSEv2Model(cfg)
        seq = torch.randint(4, cfg.vocab_size, (2, 16))
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = model(seq)
        assert torch.isfinite(out["logits"]).all()

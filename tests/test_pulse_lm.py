"""Tests for GPU-scaled PULSELanguageModel (Hybrid-Local) and Matched ScalableTransformerLM."""

import torch

from pulse.pulse_lm import PULSELMConfig, PULSELanguageModel
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM


class TestMatchedParameterScaling:
    def test_30m_experiment_presets_match_parameter_counts_closely(self):
        pcfg = PULSELMConfig.colab_30m()
        tcfg = ScalableTransformerConfig.colab_30m()
        # Instantiate on meta device so 30M parameter counting takes milliseconds and 0 RAM
        with torch.device("meta"):
            pulse_lm = PULSELanguageModel(pcfg)
            tf_lm = ScalableTransformerLM(tcfg)

        p_params = pulse_lm.num_parameters()
        t_params = tf_lm.num_parameters()
        ratio = p_params / t_params

        assert 22_000_000 <= p_params <= 40_000_000
        assert 0.85 <= ratio <= 1.15, f"Param mismatch: PULSE={p_params:,} vs TF={t_params:,}"


class TestPULSELanguageModel:
    def test_strict_causality_future_tokens_do_not_affect_past_logits(self):
        torch.manual_seed(0)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        model.eval()

        seq_a = torch.randint(4, cfg.vocab_size, (2, 16))
        seq_b = seq_a.clone()
        seq_b[:, 12:] = (seq_b[:, 12:] + 7) % cfg.vocab_size

        with torch.no_grad():
            out_a = model(seq_a)
            out_b = model(seq_b)

        # Tokens 0..11 must have identical next-token logits regardless of changes at 12..15
        assert torch.allclose(out_a["logits"][:, :11, :], out_b["logits"][:, :11, :], atol=1e-5)

    def test_hybrid_local_training_decreases_loss_with_zero_backprop_in_predictive_core(self):
        torch.manual_seed(7)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=3e-3)

        # Repeating structured token patterns
        base = torch.tensor(
            [
                [5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10, 11, 12],
                [20, 21, 22, 23, 24, 25, 26, 27, 20, 21, 22, 23, 24, 25, 26, 27],
                [30, 31, 32, 33, 34, 35, 36, 37, 30, 31, 32, 33, 34, 35, 36, 37],
                [40, 41, 42, 43, 44, 45, 46, 47, 40, 41, 42, 43, 44, 45, 46, 47],
            ],
            dtype=torch.long,
        )

        losses = []
        for _ in range(35):
            stats = model.train_step_hybrid_local(base, io_optimizer=io_opt, async_core=True)
            losses.append(stats["ce_loss"])

        # STRICT AUDIT: Predictive Coding Field & Causal Transition Core MUST have ZERO .grad!
        for name, p in model.field.named_parameters():
            assert p.grad is None, f"Core field parameter {name} leaked into global backprop!"
        for name, p in model.causal_layer.named_parameters():
            assert p.grad is None, f"Core causal parameter {name} leaked into global backprop!"
        assert model.core_autograd_grad_count() == 0

        assert losses[-1] < losses[0] * 0.5

    def test_working_state_is_strictly_o1_across_sequence_lengths(self):
        torch.manual_seed(11)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        model.eval()

        with torch.no_grad():
            short_out = model(torch.randint(4, cfg.vocab_size, (1, 16)))
            long_out = model(torch.randint(4, cfg.vocab_size, (1, 128)))

        assert short_out["working_state"].shape == long_out["working_state"].shape
        assert (
            model.working_state_bytes(short_out["working_state"])
            == model.working_state_bytes(long_out["working_state"])
        )

    def test_online_fact_teaching_zero_context_recall_and_sleep_consolidation(self):
        torch.manual_seed(19)
        cfg = PULSELMConfig.micro_test()
        cfg.surprise_threshold = 0.05
        cfg.surprise_temperature = 0.02
        cfg.fast_weight_alpha = 0.9
        cfg.fast_weight_decay = 0.0
        cfg.consolidation_lr = 0.14
        model = PULSELanguageModel(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=3e-3)

        # Pre-train on an "old" sequence
        old_cue = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
        old_target = torch.tensor([[18, 19, 20, 21]], dtype=torch.long)
        old_seq = torch.cat([old_cue, old_target], dim=1)
        for _ in range(25):
            model.train_step_hybrid_local(old_seq, io_optimizer=io_opt)

        # Teach a brand-new unseen fact ONLINE into W_fast (zero backprop!)
        new_cue = torch.tensor([[60, 61, 62, 63, 64, 65, 66, 67]], dtype=torch.long)
        new_target = torch.tensor([[80, 81, 82, 83]], dtype=torch.long)

        z_new_truth = model.encode_chunks(new_target)[:, 0, :].detach()
        z_old_truth = model.encode_chunks(old_target)[:, 0, :].detach()

        err_before = (model.predict_next_concept(new_cue) - z_new_truth).pow(2).mean().item()
        teach_stats = model.teach_fact_online(new_cue, new_target, repeats=5)
        err_after_online = (model.predict_next_concept(new_cue) - z_new_truth).pow(2).mean().item()

        assert teach_stats["max_gate"] > 0.7
        assert err_after_online < err_before * 0.25

        # Sleep consolidation: transfer W_fast -> W_slow and wipe W_fast = 0
        sleep_stats = model.sleep_consolidate(num_replay_samples=24, num_epochs=35)
        assert sleep_stats["causal_consolidated"] >= 1
        assert model.causal_layer.w_fast.norm().item() < 1e-6

        err_new_post_sleep = (
            model.predict_next_concept(new_cue, use_fast_weights=False) - z_new_truth
        ).pow(2).mean().item()
        err_old_post_sleep = (
            model.predict_next_concept(old_cue, use_fast_weights=False) - z_old_truth
        ).pow(2).mean().item()

        assert err_new_post_sleep < err_before * 0.35
        assert err_old_post_sleep < 0.45

    def test_generation_and_continuation_consistency_energy(self):
        torch.manual_seed(23)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=3e-3)

        prompt = torch.tensor([[12, 13, 14, 15, 16, 17, 18, 19]], dtype=torch.long)
        good_end = torch.tensor([[20, 21, 22, 23]], dtype=torch.long)
        bad_end = torch.tensor([[90, 91, 92, 93]], dtype=torch.long)
        full_good = torch.cat([prompt, good_end], dim=1)

        for _ in range(30):
            model.train_step_hybrid_local(full_good, io_optimizer=io_opt)

        gen = model.generate(prompt, max_new_tokens=8, temperature=0.0)
        assert gen.shape == (1, 16)

        e_good = model.continuation_free_energy(prompt, good_end).item()
        e_bad = model.continuation_free_energy(prompt, bad_end).item()
        assert e_bad > e_good * 1.5

    def test_bf16_autocast_forward_and_hybrid_step_preserve_fp32_core_stability(self):
        torch.manual_seed(31)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        io_opt = torch.optim.AdamW(model.io_parameters(), lr=2e-3)
        seq = torch.randint(4, cfg.vocab_size, (2, 16))

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            out = model(seq[:, :-1])
        assert torch.isfinite(out["logits"]).all()
        assert torch.isfinite(out["final_energy"]).all()

        stats = model.train_step_hybrid_local(seq, io_optimizer=io_opt)
        assert stats["ce_loss"] > 0.0
        assert model.core_autograd_grad_count() == 0

    def test_parallel_ssm_scan_matches_sequential_step_exactly(self):
        torch.manual_seed(42)
        cfg = PULSELMConfig.micro_test()
        model = PULSELanguageModel(cfg)
        z_seq = torch.randn(3, 17, cfg.top_dim)  # Non-power-of-two length!
        h0 = torch.randn(3, cfg.ssm_state_dim) * 0.5

        h_final_par, prior_par, h_seq_par = model._scan_fast_ssm(z_seq, init_state=h0)

        # Sequential step-by-step reference
        h = h0.clone()
        h_list = []
        for t in range(z_seq.shape[1]):
            h, _ = model.fast_ssm.step(z_seq[:, t, :], h)
            h_list.append(h)
        h_seq_ref = torch.stack(h_list, dim=1)
        prior_ref = model.fast_ssm.predict_prior(h_seq_ref)

        assert torch.allclose(h_seq_par, h_seq_ref, atol=1e-5)
        assert torch.allclose(h_final_par, h, atol=1e-5)
        assert torch.allclose(prior_par, prior_ref, atol=1e-5)


class TestScalableTransformerLM:
    def test_forward_kv_cache_and_generation_work_cleanly(self):
        torch.manual_seed(0)
        cfg = ScalableTransformerConfig.micro_test()
        tf = ScalableTransformerLM(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 16))

        logits, kv = tf(tokens, use_cache=True)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert len(kv) == cfg.n_layers

        kv_16 = tf.kv_cache_bytes(kv)
        _, kv_64 = tf(torch.randint(0, cfg.vocab_size, (2, 64)), use_cache=True)
        assert tf.kv_cache_bytes(kv_64) == kv_16 * 4

        gen = tf.generate(tokens[:1, :8], max_new_tokens=6, temperature=0.0)
        assert gen.shape == (1, 14)

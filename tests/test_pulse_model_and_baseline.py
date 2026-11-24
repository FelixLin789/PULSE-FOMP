"""End-to-end integration tests for PULSEModel, PulseTokenizer, and TransformerBaseline."""

import pytest
import torch

from pulse.baseline_transformer import TransformerBaseline, TransformerConfig
from pulse.config import PULSEConfig
from pulse.model import PULSEModel
from pulse.tokenizer import PulseTokenizer


class TestPulseTokenizer:
    def test_roundtrips_english_chinese_and_code_text(self):
        tok = PulseTokenizer()
        samples = [
            "PULSE latent dynamics!",
            "可塑性隐态动力学引擎",
            "def f(x): return x * 2",
        ]
        for text in samples:
            ids = tok.encode(text, max_len=64)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_batch_encode_pads_uniformly(self):
        tok = PulseTokenizer()
        batch = tok.encode_batch(["hi", "hello world"], max_len=16)
        assert batch.shape == (2, 16)
        assert batch[0, -1].item() == tok.PAD_ID


class TestPULSEModelEndToEnd:
    def test_think_and_forward_handle_long_sequences_with_o1_working_memory(self):
        torch.manual_seed(0)
        cfg = PULSEConfig.micro()
        cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
        cfg.output_dim = PulseTokenizer.VOCAB_SIZE
        model = PULSEModel(cfg)

        tok = PulseTokenizer()
        seq_short = tok.encode_batch(["alpha beta gamma delta"], max_len=16)
        seq_long = tok.encode_batch(["alpha beta " * 40], max_len=256)

        out_short = model.process_sequence(seq_short)
        out_long = model.process_sequence(seq_long)

        # Working state memory footprint is strictly identical regardless of sequence length
        assert out_short["working_state"].shape == out_long["working_state"].shape
        assert out_short["working_state"].nelement() == out_long["working_state"].nelement()
        assert torch.isfinite(out_long["working_state"]).all()

    def test_pure_local_training_improves_causal_prediction_with_zero_backprop(self):
        torch.manual_seed(11)
        cfg = PULSEConfig.micro()
        cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
        cfg.output_dim = 8
        cfg.weight_lr = 0.03
        cfg.io_lr = 0.04
        model = PULSEModel(cfg)

        tok = PulseTokenizer()
        cues = tok.encode_batch(
            [f"rule_{i}: input_signal_{i}" for i in range(8)], max_len=24
        )
        targets = torch.arange(8)

        with torch.no_grad():
            acc_before = (model(cues)["logits"].argmax(dim=-1) == targets).float().mean().item()

        for _ in range(50):
            model.train_step_local(cues, target_class=targets)

        # Zero autograd gradients anywhere in the entire PULSE model
        for name, param in model.named_parameters():
            assert param.grad is None, f"Unexpected autograd .grad on {name}!"

        with torch.no_grad():
            acc_after = (model(cues)["logits"].argmax(dim=-1) == targets).float().mean().item()

        assert acc_after >= 0.875
        assert acc_after > acc_before

    def test_inference_time_online_teaching_and_zero_context_recall_plus_sleep(self):
        torch.manual_seed(17)
        cfg = PULSEConfig.micro()
        cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
        cfg.output_dim = PulseTokenizer.VOCAB_SIZE
        cfg.surprise_threshold = 0.05
        cfg.surprise_temperature = 0.02
        cfg.fast_weight_alpha = 0.9
        cfg.fast_weight_decay = 0.0
        cfg.consolidation_lr = 0.15
        cfg.consolidation_ewc_lambda = 0.15
        model = PULSEModel(cfg)
        tok = PulseTokenizer()

        # 1. Pre-establish an "old" rule in W_slow via local training
        old_cue = tok.encode_batch(["old_law: gravity"], max_len=24)
        old_ans = tok.encode_batch(["falls_down"], max_len=24)
        for _ in range(35):
            model.train_causal_concept_local(old_cue, old_ans)

        # 2. Teach a brand-new private rule ONLINE during inference (into W_fast, zero backprop!)
        new_cue = tok.encode_batch(["private_code: ZETA9"], max_len=24)
        new_ans = tok.encode_batch(["override_alpha"], max_len=24)
        z_new_target = model.encode_text(new_ans).detach()
        z_old_target = model.encode_text(old_ans).detach()

        err_before = (model.predict_causal_concept(new_cue) - z_new_target).pow(2).mean().item()
        teach_stats = model.teach_online(new_cue, new_ans, repeats=4)

        # 3. Query with ZERO working state / ZERO context window -> recall from W_fast!
        model.reset_working_state()
        err_after_online = (model.predict_causal_concept(new_cue) - z_new_target).pow(2).mean().item()
        assert teach_stats["max_gate"] > 0.7
        assert err_after_online < err_before * 0.25

        # 4. Run Sleep Consolidation -> wipes W_fast to 0 and distills into W_slow!
        sleep_stats = model.sleep_consolidate(num_replay_samples=24, num_epochs=35)
        assert sleep_stats["causal_consolidated"] >= 1
        assert model.causal_layer.w_fast.norm().item() < 1e-6

        # 5. Verify BOTH new rule AND old rule survive purely in W_slow!
        err_new_after_sleep = (
            model.predict_causal_concept(new_cue, use_fast_weights=False) - z_new_target
        ).pow(2).mean().item()
        err_old_after_sleep = (
            model.predict_causal_concept(old_cue, use_fast_weights=False) - z_old_target
        ).pow(2).mean().item()

        assert err_new_after_sleep < err_before * 0.35
        assert err_old_after_sleep < 0.35

    def test_consistency_check_distinguishes_learned_truth_from_hallucination(self):
        torch.manual_seed(29)
        cfg = PULSEConfig.micro()
        cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
        cfg.output_dim = PulseTokenizer.VOCAB_SIZE
        model = PULSEModel(cfg)
        tok = PulseTokenizer()

        premise = tok.encode_batch(["water boils at"], max_len=20)
        true_ans = tok.encode_batch(["100 celsius"], max_len=20)
        false_ans = tok.encode_batch(["minus 999 kelvin"], max_len=20)

        for _ in range(40):
            model.train_causal_concept_local(premise, true_ans)

        e_true = model.check_consistency(premise, true_ans)["consistency_energy"].item()
        e_false = model.check_consistency(premise, false_ans)["consistency_energy"].item()

        assert e_false > e_true * 3.0

    def test_save_and_load_preserves_slow_and_fast_weights(self, tmp_path):
        torch.manual_seed(31)
        cfg = PULSEConfig.micro()
        cfg.vocab_size = PulseTokenizer.VOCAB_SIZE
        model = PULSEModel(cfg)
        tok = PulseTokenizer()

        cue = tok.encode_batch(["secret_key"], max_len=16)
        ans = tok.encode_batch(["secret_val"], max_len=16)
        model.teach_online(cue, ans, repeats=2)
        pred_before = model.predict_causal_concept(cue)

        ckpt = tmp_path / "pulse_test.pt"
        model.save_checkpoint(str(ckpt))
        loaded = PULSEModel.load_checkpoint(str(ckpt))
        pred_after = loaded.predict_causal_concept(cue)

        assert torch.allclose(pred_before, pred_after, atol=1e-6)


class TestTransformerBaseline:
    def test_causal_transformer_forward_and_kv_cache_growth(self):
        torch.manual_seed(0)
        tcfg = TransformerConfig(
            vocab_size=PulseTokenizer.VOCAB_SIZE,
            d_model=48,
            n_heads=4,
            n_layers=2,
            d_ff=96,
            max_seq_len=512,
        )
        tf = TransformerBaseline(tcfg)
        tokens = torch.randint(0, tcfg.vocab_size, (2, 32))

        logits, kv_cache = tf(tokens, use_cache=True)
        assert logits.shape == (2, 32, tcfg.vocab_size)
        assert len(kv_cache) == tcfg.n_layers
        cache_bytes_32 = tf.kv_cache_bytes(kv_cache)

        tokens_long = torch.randint(0, tcfg.vocab_size, (2, 128))
        _, kv_cache_long = tf(tokens_long, use_cache=True)
        cache_bytes_128 = tf.kv_cache_bytes(kv_cache_long)

        # Transformer KV cache grows linearly 4x with 4x sequence length
        assert cache_bytes_128 == cache_bytes_32 * 4

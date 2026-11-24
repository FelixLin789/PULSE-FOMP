from dataclasses import replace

import torch

from pulse.pulse_v4 import PULSEv4Config, PULSEv4Model
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model


def assert_compact_state(model, state):
    allocations = {}
    for layer in [*state["blocks"], state]:
        for value in layer.values():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                allocations[storage.data_ptr()] = storage.nbytes()
    assert sum(allocations.values()) == model.working_state_bytes(state)


def test_v6_migration_and_compact_multitoken_streaming():
    torch.manual_seed(17)
    legacy = PULSEv4Model(PULSEv4Config.micro_test()).eval()
    compat_cfg = replace(PULSEv6Config.micro_test(), use_attn_res=True, delta_beta_floor=0.05)
    model = PULSEv6Model(compat_cfg).eval()
    with torch.no_grad():
        legacy.emb_out.weight.normal_(std=.06)
        for block in legacy.blocks:
            block.o_proj.weight.normal_(std=.02)
            block.w_down.weight.normal_(std=.02)
        model.load_v53_backbone(legacy.state_dict())
        tokens = torch.randint(4, 260, (2, 160))
        original = legacy(tokens, run_core=False, return_state=False)["logits"]
        migrated = model(tokens, return_state=False)["logits"]
        torch.testing.assert_close(original, migrated, rtol=0, atol=0)
        whole = model(tokens)["logits"]
        assert original.abs().max() > 1
        torch.testing.assert_close(migrated, whole, rtol=1e-5, atol=3e-6)
        state = None
        parts = []
        for chunk in torch.split(tokens, (64, 31, 64, 1), dim=1):
            out = model(chunk, state)
            parts.append(out["logits"])
            state = out["working_state"]
            assert_compact_state(model, state)
        torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=1e-5, atol=5e-6)
    assert all(name not in dict(model.named_modules()) for name in ("field", "causal_layer", "contrastive_critic"))


def test_rls_replay_exact_edits_survive_sleep_and_checkpoint():
    torch.manual_seed(39)
    config = replace(PULSEv6Config.micro_test(), memory_capacity=4)
    model = PULSEv6Model(config).eval()
    with torch.no_grad():
        model.emb_out.weight.normal_(std=.07)
        short = torch.tensor([[14, 15, 16]])
        cues = torch.tensor([[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]])
        targets = torch.tensor([87, 94])
        text = torch.tensor([[23, 15, 16, 14, 17, 16, 19, 20]])
        before = model(text)["logits"]
        model.teach_fact_online(short, torch.tensor([97]))
        model.teach_fact_online(cues, targets)
        first = model(cues[:, :4])["working_state"]
        streamed = model(cues[:, 4:], first, last_logits_only=True)["logits"][:, 0]
        assert streamed.argmax(-1).equal(targets)
        assert model(short)["logits"][0, -1].argmax().item() == 97
        torch.testing.assert_close(model(text)["logits"], before, rtol=0, atol=0)

        updated = torch.tensor([123])
        assert model.sleep_consolidate()["consolidated_facts"] == 3
        model.teach_fact_online(cues[:1], updated)
        assert model(cues[:1])["logits"][0, -1].argmax().item() == 123
        assert model(cues[:1], use_fast_weights=False)["logits"][0, -1].argmax().item() == 87
        restored = PULSEv6Model(config).eval()
        restored.load_state_dict(model.state_dict())
        assert restored.sleep_consolidate()["consolidated_facts"] == 1
        assert restored.memory.fast.count_nonzero() == 0
        assert restored(cues, use_fast_weights=False)["logits"][:, -1].argmax(-1).tolist() == [123, 94]
        torch.testing.assert_close(restored(text)["logits"], before, rtol=0, atol=0)
        generated = restored.generate(cues, 1, temperature=0, use_fast_weights=False)
        assert generated[:, -1].tolist() == [123, 94]


def test_plain_pretraining_does_not_read_or_update_memory():
    torch.manual_seed(8)
    model = PULSEv6Model(PULSEv6Config.micro_test()).eval()
    tokens = torch.randint(4, 260, (2, 20))
    model.teach_fact_online(tokens[:, :16], torch.tensor([87, 94]))
    no_memory = PULSEv6Model(replace(model.config, memory_capacity=0))
    no_memory.load_v53_backbone(model.state_dict())
    torch.testing.assert_close(model.compute_loss(tokens), no_memory.compute_loss(tokens), rtol=0, atol=0)
    slow = model.memory.slow.clone()
    fast = model.memory.fast.clone()
    model.compute_loss(tokens).backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.equal(model.memory.slow, slow) and torch.equal(model.memory.fast, fast)


def test_edit_corrects_competitors_with_higher_norm_embeddings():
    torch.manual_seed(14)
    model = PULSEv6Model(PULSEv6Config.micro_test()).eval()
    with torch.no_grad():
        model.emb_out.weight.normal_(std=.1)
        # A target-only correction raises its correlated competitor even faster.
        model.tok_emb.weight[259].zero_()
        model.tok_emb.weight[259, 0] = 1.
        model.tok_emb.weight[258].zero_()
        model.tok_emb.weight[258, :2] = torch.tensor([2., 1.])
        cue = torch.tensor([[14, 17, 26, 32]])
        score = model.teach_fact_online(cue, torch.tensor([259]))
        assert score["recall"] == 1.
        model.sleep_consolidate()
        assert model(cue, use_fast_weights=False)["logits"][0, -1].argmax().item() == 259


def test_v6_defaults_no_attnres_and_zero_beta_floor_stability():
    for cfg in (PULSEv6Config.micro_test(), PULSEv6Config.colab_30m(), PULSEv6Config.colab_100m()):
        assert cfg.use_attn_res is False
        assert cfg.delta_beta_floor == 0.0

    torch.manual_seed(23)
    model = PULSEv6Model(PULSEv6Config.micro_test())
    with torch.no_grad():
        model.emb_out.weight.normal_(std=.06)
        for blk in model.blocks:
            blk.o_proj.weight.normal_(std=.02)
            blk.w_down.weight.normal_(std=.02)
        # Extreme negative erase bias: beta -> 0 must stay finite in both FP32 and BF16 (no div-by-zero).
        b0 = model.blocks[0]
        h = b0.n_heads
        b0.gates_proj.bias[h : 2 * h].fill_(-20.0)
        b0.gates_proj.bias[:h].fill_(-15.0)

    tokens = torch.randint(4, 260, (2, 33))
    loss = model.compute_loss(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    model.eval()
    with torch.no_grad():
        whole = model(tokens)["logits"]
        state = None
        parts = []
        for t in range(tokens.size(1)):
            out = model(tokens[:, t : t + 1], state)
            parts.append(out["logits"])
            state = out["working_state"]
            assert_compact_state(model, state)
        torch.testing.assert_close(torch.cat(parts, dim=1), whole, rtol=1e-4, atol=1e-4)
        # Under beta -> 0, state norm stays near zero (pure selective no-op skip).
        assert state["blocks"][0]["S"].abs().max().item() < 1e-4


def test_mega_stage_topology_6l_and_12l_exact_parity():
    assert not PULSEv6Model(PULSEv6Config.micro_test())._has_mega_stage_topology()
    for n_layers, deltas in ((6, (0, 3)), (12, (0, 3, 6, 9))):
        for use_ar in (False, True):
            torch.manual_seed(100 + n_layers + int(use_ar))
            cfg = PULSEv6Config(
                vocab_size=128, embed_dim=24, d_model=48, n_heads=4, n_kv_heads=2,
                n_layers=n_layers, delta_layers=deltas, d_ff=96, delta_chunk_size=4,
                echoic_window_size=8, use_attn_res=use_ar, memory_capacity=0,
            )
            model = PULSEv6Model(cfg).eval()
            assert model._has_mega_stage_topology()
            with torch.no_grad():
                model.emb_out.weight.normal_(std=.05)
                for blk in model.blocks:
                    blk.o_proj.weight.normal_(std=.03)
                    blk.w_down.weight.normal_(std=.03)

                tokens = torch.randint(4, 128, (2, 17))

                def cpu_delta_runner(blk, q, k, v_g, la, be, ret_st):
                    S0 = torch.zeros(q.size(0), blk.n_heads, blk.head_dim, blk.head_dim, dtype=torch.float32)
                    o, sf, _ = blk._forward_wy_cpu(q, k, v_g * be.unsqueeze(-1), la, be, S0, False)
                    return o, (sf if ret_st else None)

                for ret_st in (False, True):
                    ref_fin, ref_emb, ref_mid, ref_sts, _ = model._run_backbone(tokens, return_state=ret_st)
                    stages = model._make_mega_stage_fns(emit_state=ret_st)
                    assert len({fn.__code__ for fn in (stages[0], *stages[1], stages[2])}) == len(stages[1]) + 2
                    mg_fin, mg_emb, mg_mid, mg_sts, _ = model._run_mega_stages(
                        tokens, stages, cpu_delta_runner, return_state=ret_st
                    )
                    torch.testing.assert_close(mg_fin, ref_fin, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(mg_emb, ref_emb, rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(mg_mid, ref_mid, rtol=1e-5, atol=1e-5)
                    assert len(mg_sts) == (n_layers if ret_st else 0)
                    if ret_st:
                        assert_compact_state(model, {"blocks": mg_sts})
                        for st_m, st_r in zip(mg_sts, ref_sts, strict=True):
                            assert st_m.keys() == st_r.keys()
                            for k_name in st_m:
                                if isinstance(st_m[k_name], torch.Tensor):
                                    torch.testing.assert_close(st_m[k_name], st_r[k_name], rtol=1e-5, atol=1e-5)
                                else:
                                    assert st_m[k_name] == st_r[k_name]

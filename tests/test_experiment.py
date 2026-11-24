"""Tests for the 200-CCU Colab Controlled Experiment Runner (PULSE-LM vs. Transformer)."""

from pathlib import Path
import json

from pulse.experiment import (
    CCUBudgetGuard,
    ControlledExperimentRunner,
    ExperimentConfig,
)


class TestCCUBudgetGuard:
    def test_estimates_gpu_rates_and_enforces_phase_budget(self):
        guard = CCUBudgetGuard(total_ccu_budget=180.0, gpu_name="NVIDIA A100-SXM4-40GB")
        assert guard.ccu_per_hour >= 10.0
        assert guard.remaining_ccu() == 180.0
        assert not guard.phase_exceeded("pulse", phase_fraction=0.45)

        # Simulate spending 85 CCU on phase A (> 45% of 180 = 81 CCU)
        guard.record_spent_ccu("pulse", 85.0)
        assert guard.phase_exceeded("pulse", phase_fraction=0.45)
        assert not guard.phase_exceeded("transformer", phase_fraction=0.45)
        assert abs(guard.remaining_ccu() - 95.0) < 1e-6


class TestControlledExperimentRunner:
    def test_end_to_end_training_checkpoint_resume_and_six_axis_report(self, tmp_path: Path):
        out_dir = tmp_path / "exp_run"
        cfg = ExperimentConfig.micro_smoke(output_dir=str(out_dir))
        cfg.max_steps = 6
        cfg.checkpoint_every = 3

        # Run 1: stop partway (3 steps) to verify checkpointing & resume
        cfg_part1 = ExperimentConfig.micro_smoke(output_dir=str(out_dir))
        cfg_part1.max_steps = 3
        cfg_part1.checkpoint_every = 3
        runner1 = ControlledExperimentRunner(cfg_part1)
        res1 = runner1.run_training_phase("pulse")
        assert res1["steps_completed"] == 3
        assert (out_dir / "checkpoints" / "pulse_latest.pt").exists()

        # Run 2: resume seamlessly to step 6, train matched Transformer, and run full 6-axis eval
        runner2 = ControlledExperimentRunner(cfg)
        full_results = runner2.run_all()

        assert full_results["pulse_training"]["steps_completed"] == 6
        assert full_results["transformer_training"]["steps_completed"] == 6
        assert full_results["six_axis_eval"]["axis6_local_learning"]["pulse_core_autograd_grads"] == 0
        assert (out_dir / "experiment_results.json").exists()
        assert (out_dir / "EXPERIMENT_REPORT.md").exists()
        assert (out_dir / "training_curves.png").exists()
        assert (out_dir / "six_axis_comparison.png").exists()

        saved = json.loads((out_dir / "experiment_results.json").read_text(encoding="utf-8"))
        assert "six_axis_eval" in saved
        ev = saved["six_axis_eval"]
        assert ev["axis1_complexity_memory"]["pulse_state_kb"][-1] < (
            ev["axis1_complexity_memory"]["transformer_kv_kb"][-1]
        )
        # Verify Step 1 rigor upgrades are present
        assert "pulse_online_base_ppl_drift_pct" in ev["axis3_online_plasticity"]
        assert "hard_entity_swap_auroc" in ev["axis5_anti_hallucination"]
        assert "pulse_extrap_tail_ppl" in ev["axis7_length_extrapolation"]
        assert "non_repeating_verified" in ev["axis7_length_extrapolation"]
        assert "little fox" not in ev["sample_generations"]["prompt"]

    def test_fair_muon_optimizer_runs_cleanly_for_both_models(self, tmp_path: Path):
        out_dir = tmp_path / "muon_smoke"
        cfg = ExperimentConfig.micro_smoke(output_dir=str(out_dir))
        cfg.optimizer_type = "muon"
        cfg.max_steps = 2
        cfg.checkpoint_every = 2
        runner = ControlledExperimentRunner(cfg)
        res = runner.run_all()
        assert res["pulse_training"]["steps_completed"] == 2
        assert res["transformer_training"]["steps_completed"] == 2

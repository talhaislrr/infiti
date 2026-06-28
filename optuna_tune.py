"""
Optuna ile HFP hiperparametre optimizasyonu
===========================================
Kullanım:
  pip install optuna
  python3 optuna_tune.py --quick
"""

from __future__ import annotations

import argparse

import torch

from hfp_config import HFPConfig, optuna_objective, train_with_hfp, build_hfp_model
from benchmark import make_mnist_loaders, set_seed


def main():
    try:
        import optuna
    except ImportError:
        print("Optuna gerekli: pip install optuna")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.quick:
        args.trials = 5
        args.epochs = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    train_l, val_l, test_l = make_mnist_loaders(seed=42)

    study = optuna.create_study(direction="minimize", study_name="hfp_mnist")
    study.optimize(
        lambda trial: optuna_objective(trial, train_l, val_l, test_l, device, args.epochs),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print("\n── En iyi trial ──")
    print(f"  Val acc: {-study.best_value*100:.2f}%")
    print(f"  Params: {study.best_params}")

    best = study.best_params
    best_config = HFPConfig(
        bulk_rank=best["bulk_rank"],
        stiffness_p=best["stiffness_p"],
        stiffness_threshold=best["stiffness_threshold"],
        zenon_schedule_points=[best["zenon_p1"], best["zenon_p2"]],
        zenon_grad_threshold=best["zenon_grad_threshold"],
        initial_lr=best["initial_lr"],
        use_zenon=best["use_zenon"],
        max_epochs=args.epochs,
        use_bulk=True,
        use_stiff=True,
    )
    best_config.to_json("hfp_config_optimized.json")
    print("  Kaydedildi: hfp_config_optimized.json")

    model = build_hfp_model(784, 512, 10, best_config)
    result = train_with_hfp(model, best_config, train_l, val_l, test_l, device)
    print(f"  Test acc: {result.test_acc*100:.2f}% | {result.elapsed_sec}s | {result.epochs_run} epoch")


if __name__ == "__main__":
    main()

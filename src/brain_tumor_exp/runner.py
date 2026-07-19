from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, RuntimeConfig
from .train_eval import train_one_experiment
from .utils import save_json, set_seed


def build_experiment_grid(
    model_names: list[str],
    image_sizes: list[int],
    preprocessing_flags: list[bool],
    pretrained_flags: list[bool],
    use_augmentation: bool,
) -> list[ExperimentConfig]:
    experiments: list[ExperimentConfig] = []
    for model_name in model_names:
        for image_size in image_sizes:
            for use_preprocessing in preprocessing_flags:
                for use_pretrained in pretrained_flags:
                    if model_name in {"simple_cnn", "simple_cnn_deeper"} and use_pretrained:
                        continue
                    experiments.append(
                        ExperimentConfig(
                            model_name=model_name,
                            image_size=image_size,
                            use_preprocessing=use_preprocessing,
                            use_pretrained=use_pretrained,
                            use_augmentation=use_augmentation,
                        )
                    )
    return experiments


def run_experiments(runtime_cfg: RuntimeConfig, experiments: list[ExperimentConfig]) -> pd.DataFrame:
    set_seed(runtime_cfg.seed)
    runtime_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, exp_cfg in enumerate(experiments, start=1):
        print(f"\n[{index}/{len(experiments)}] Running {exp_cfg.experiment_id}")
        metrics = train_one_experiment(runtime_cfg=runtime_cfg, exp_cfg=exp_cfg)

        row = {
            "experiment_id": exp_cfg.experiment_id,
            **asdict(exp_cfg),
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
            "checkpoint": metrics["checkpoint"],
        }
        rows.append(row)

        per_exp_path = runtime_cfg.output_dir / "metrics" / f"{exp_cfg.experiment_id}.json"
        save_json(
            {
                "experiment": asdict(exp_cfg),
                "metrics": metrics,
            },
            per_exp_path,
        )

    results_df = pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)
    csv_path = runtime_cfg.output_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)

    summary_path = runtime_cfg.output_dir / "summary.json"
    save_json(
        {
            "runtime": {
                "project_root": str(runtime_cfg.project_root),
                "dataset_dir": str(runtime_cfg.dataset_dir),
                "output_dir": str(runtime_cfg.output_dir),
                "batch_size": runtime_cfg.batch_size,
                "epochs": runtime_cfg.epochs,
                "learning_rate": runtime_cfg.learning_rate,
                "weight_decay": runtime_cfg.weight_decay,
                "num_workers": runtime_cfg.num_workers,
                "seed": runtime_cfg.seed,
                "device": runtime_cfg.device,
            },
            "best_experiment": results_df.iloc[0].to_dict() if not results_df.empty else None,
            "results_csv": str(csv_path),
        },
        summary_path,
    )
    return results_df

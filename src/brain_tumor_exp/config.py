from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    dataset_dir: Path
    output_dir: Path
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    image_size: int
    use_preprocessing: bool
    use_pretrained: bool
    use_augmentation: bool = True

    @property
    def experiment_id(self) -> str:
        preprocess_tag = "prep" if self.use_preprocessing else "raw"
        pretrained_tag = "pt" if self.use_pretrained else "scratch"
        aug_tag = "aug" if self.use_augmentation else "noaug"
        return f"{self.model_name}_{self.image_size}_{preprocess_tag}_{pretrained_tag}_{aug_tag}"

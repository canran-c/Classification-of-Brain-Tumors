Developing Effective AI-driven Approaches for Classification of Brain Tumors in Medical Imaging

Abstract: Brain tumor diagnosis is of critical importance in clinical practice. Traditional MRI-based manual diagnosis is often time-consuming, highly dependent on expert radiologists, and subject
to considerable uncertainty. During the past decades, artificial intelligence (AI), particularly deep learning, has achieved remarkable success across various domains including medical diagnosis. Motivated
by these advances, this study investigates AI-driven approaches for brain tumor classification. Building upon an extensive review of deep learning literature and a thorough understanding
of convolutional neural network (CNN) architectures, I employ multiple CNNs to address the brain tumor classification task. Experiments are conducted on a benchmark dataset to evaluate
and compare the classification performance of different models. In addition, the effectiveness of image preprocessing and model pretraining is explored. Based on the experimental findings, I
further analyze the limitations of current CNN-based classification methods and discuss potential future research directions for improving brain tumor diagnosis.

# Brain Tumor Classification Experiments Report

This report summarizes three experiment blocks:

1) Hyperparameter search for the main setting (`resnet18`, image size `128`, with preprocessing, with ImageNet pretraining), using approximate grid search over learning rate and epochs.

2) Full requirement-oriented experiments using four CNNs and two image scales, with/without preprocessing and with/without pretraining where applicable.

3) An additional single-GPU experiment block for `vgg11` and `efficientnet_b0`, also evaluated over two image sizes and with/without preprocessing and pretraining.


## 1. Experiment Design

### 1.1 Hyperparameter Search Block

Fixed setting:

- Model: `resnet18`
- Image size: `128`
- Preprocessing: `True`
- Pretraining: `True`

Grid searched:

- Epochs in {10, 20, 40}
- Learning rates in {1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5} (partially covered by available runs)

Executed via [scripts/r18_imgsz128_pretr_preproc_gridsrch.sh](scripts/r18_imgsz128_pretr_preproc_gridsrch.sh#L1-L24).

Total runs: 13.

### 1.2 Full Experiment Block

Executed via [scripts/4nets_2imgszs_wwo-pretr_wwo-preproc.sh](scripts/4nets_2imgszs_wwo-pretr_wwo-preproc.sh#L1-L36), split by preprocessing flag across two GPUs:

- GPU0: preprocessing `true` ([L10-L21](scripts/4nets_2imgszs_wwo-pretr_wwo-preproc.sh#L10-L21))
- GPU1: preprocessing `false` ([L24-L35](scripts/4nets_2imgszs_wwo-pretr_wwo-preproc.sh#L24-L35))

Run outputs:

- `results0.csv` + `summary0.json` (GPU0)
- `results1.csv` + `summary1.json` (GPU1)

After merge/deduplication, valid runs: **24**.

Note: the nominal Cartesian size `4 × 2 × 2 × 2 = 32` is reduced to 24 because custom CNNs (`simple_cnn`, `simple_cnn_deeper`) are scratch-only in the code ([src/brain_tumor_exp/runner.py](src/brain_tumor_exp/runner.py#L24-L25)).

### 1.3 Additional VGG/EfficientNet Block

Executed via [scripts/2nets_2imgszs_wwo-pretr_wwo-preproc.sh](scripts/2nets_2imgszs_wwo-pretr_wwo-preproc.sh#L1-L14) on a single GPU.

Covered configurations:

- Models: `vgg11`, `efficientnet_b0`
- Image sizes: `128`, `224`
- Preprocessing: `True`, `False`
- Pretraining: `True`, `False`

Total runs: **16**.

---

## 2. Experimental Results

### 2.1 Hyperparameter Search (Main Setting)

Top configurations by accuracy:

| Rank | Epoch |   LR | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) |
| ---- | ----: | ---: | -------: | ----------------: | -------------: | ---------: | --------------: |
| 1    |    20 | 3e-4 |   0.8046 |            0.8582 |         0.8023 |     0.7843 |          0.9516 |
| 2    |    40 | 1e-3 |   0.8046 |            0.8555 |         0.8072 |     0.7797 |          0.9461 |
| 3    |    40 | 1e-4 |   0.7995 |            0.8700 |         0.7981 |     0.7697 |          0.9469 |
| 4    |    20 | 1e-4 |   0.7970 |            0.8636 |         0.7977 |     0.7630 |          0.9487 |
| 5    |    10 | 3e-4 |   0.7919 |            0.8703 |         0.7917 |     0.7620 |          0.9142 |

Conclusion for this block: `epoch=20`, `lr=3e-4` is a strong cost-performance choice and matches your selection.

### 2.2 Overall Best Results After Adding the New 2-Model Block

Best run after adding the `vgg11`/`efficientnet_b0` experiment block:

- `efficientnet_b0_224_raw_pt_aug`
- Accuracy: **0.8401**, ROC-AUC: **0.9637**

The additional block shows that EfficientNet-B0 surpassed the previous best ResNet-based result under the same training budget (`epoch=20`, `lr=3e-4`). See [outputs_2nets-exps_e20_lr3e-4/results.csv](outputs_2nets-exps_e20_lr3e-4/results.csv#L1-L8).

### 2.3 Best Configuration per Model

| Model             | Best Experiment ID                    | Accuracy | ROC-AUC |
| ----------------- | ------------------------------------- | -------: | ------: |
| efficientnet_b0   | efficientnet_b0_224_raw_pt_aug        |   0.8401 |  0.9637 |
| resnet50          | resnet50_224_prep_pt_aug              |   0.8223 |  0.9498 |
| vgg11             | vgg11_224_prep_pt_aug                 |   0.8096 |  0.9517 |
| resnet18          | resnet18_128_prep_pt_aug              |   0.8046 |  0.9516 |
| simple_cnn_deeper | simple_cnn_deeper_128_raw_scratch_aug |   0.6574 |  0.8811 |
| simple_cnn        | simple_cnn_128_prep_scratch_aug       |   0.5761 |  0.8051 |

Sources: [outputs_full-exps_e20_lr3e-4/analysis_best_per_model.csv](outputs_full-exps_e20_lr3e-4/analysis_best_per_model.csv#L1-L5) and [outputs_2nets-exps_e20_lr3e-4/results.csv](outputs_2nets-exps_e20_lr3e-4/results.csv#L1-L16)

### 2.4 Aggregated Effects (Full Block)

#### A) Pretraining effect (ResNet-only, fair comparison)

| use_pretrained |    n | mean accuracy | max accuracy |
| -------------- | ---: | ------------: | -----------: |
| False          |    8 |        0.7500 |       0.7843 |
| True           |    8 |        0.8023 |       0.8223 |

Strong positive impact from ImageNet pretraining under a fair model family comparison (resnet18/resnet50 only; simple CNN variants excluded because they do not support pretraining).

#### B) Preprocessing effect (all models)

| use_preprocessing |    n | mean accuracy | max accuracy |
| ----------------- | ---: | ------------: | -----------: |
| False             |   12 |        0.7115 |       0.8147 |
| True              |   12 |        0.7155 |       0.8223 |

Global mean gain is small (+0.0040), but best-achievable score improves with preprocessing.

#### C) Model family summary (fair comparison under scratch-only setting)

| Model             |    n | mean accuracy | best accuracy | mean ROC-AUC |
| ----------------- | ---: | ------------: | ------------: | -----------: |
| resnet18          |    4 |        0.7659 |        0.7843 |       0.9061 |
| resnet50          |    4 |        0.7341 |        0.7513 |       0.9025 |
| simple_cnn_deeper |    4 |        0.6237 |        0.6574 |       0.8555 |
| simple_cnn        |    4 |        0.5527 |        0.5761 |       0.7924 |

Under fair scratch-only comparison, ResNet models still clearly outperform custom shallow/deeper CNN baselines, and `resnet18` is stronger than `resnet50` in this specific training budget.

### 2.5 CAM Analysis on Correct vs Wrong Predictions

To provide interpretability examples for this classification homework, Grad-CAM was run on CPU using the checkpoint:

- `resnet50_224_prep_pt_aug`
- Accuracy on test split re-check: **0.8223**

Generated artifacts:

- Example list: [outputs/cam_examples_resnet50_224_prep_pt/selected_examples.csv](outputs/cam_examples_resnet50_224_prep_pt/selected_examples.csv)
- CAM statistics: [outputs/cam_examples_resnet50_224_prep_pt/cam_stats.csv](outputs/cam_examples_resnet50_224_prep_pt/cam_stats.csv)
- Metadata: [outputs/cam_examples_resnet50_224_prep_pt/summary.json](outputs/cam_examples_resnet50_224_prep_pt/summary.json)

Selected examples include several correctly classified and misclassified samples:

- Correct examples (all predicted as `meningioma_tumor` correctly):
  - `correct_01_image(109)` / `correct_02_image(120)` / `correct_03_image(15)` / `correct_04_image(29)`
- Wrong examples (true `glioma_tumor`, predicted `meningioma_tumor`):
  - `wrong_01_image(48)` / `wrong_02_image(27)` / `wrong_03_image(89)` / `wrong_04_image(41)`

Figure A (correctly classified examples, each tile: left=input, right=Grad-CAM):

<img width="1512" height="1080" alt="panel_correct_examples" src="https://github.com/user-attachments/assets/535eb719-8e54-46f8-a1c3-8f524a9e67cb" />

Figure B (misclassified examples, each tile: left=input, right=Grad-CAM):

<img width="1512" height="1080" alt="image" src="https://github.com/user-attachments/assets/26b95462-ab6f-44cf-8efb-2abf10b93761" />

Main CAM observations:

1. ** Highly concentrated attention regions.**
   In both Figure A and Figure B, Grad-CAM activation areas are small and focused (high-activation area ratio mostly around 0.02-0.06 at threshold 0.60).

2. ** Wrong cases show similar attention geometry to correct meningioma cases.**
   The misclassified `glioma -> meningioma` examples in Figure B have CAM centroids/peaks in similar coarse regions as the correct `meningioma` examples in Figure A, indicating that the model is using overlapping visual cues.

3. ** Over-confident failure mode.**
   Selected wrong examples still have very high confidence (~0.9999-1.0000), suggesting poor confidence calibration for hard `glioma` cases.

4. ** Error pattern consistency with confusion counts.**
   For this checkpoint, the dominant confusion is `glioma_tumor -> meningioma_tumor` (47 cases), which aligns with the selected CAM failure examples.

Interpretation:

- The model has learned discriminative cues for `meningioma_tumor`, but those cues are not sufficiently specific to separate all `glioma_tumor` variants.
- A practical next step is to add class-targeted augmentation or loss reweighting for `glioma_tumor`, and evaluate calibration (e.g., temperature scaling) in addition to top-1 accuracy.

---

## 3. Discussion

1. **Pretraining is the strongest ablation factor.**
   Under fair ResNet-only comparison, average accuracy increases from 0.7500 to 0.8023 when pretraining is enabled.

2. **Preprocessing provides a modest but useful improvement ceiling.**
   While mean improvement is small overall, the best single model result uses preprocessing (`resnet50_224_prep_pt_aug`).

3. **Model capacity trade-off:**
   - With pretraining enabled, `resnet50` achieves the best absolute score.
   - Under fair scratch-only comparison, `resnet18` outperforms `resnet50` at this epoch/lr budget.
   - `simple_cnn_deeper` improves over `simple_cnn`, but both remain notably below ResNet variants.

4. **Image size effect is mild on average for ResNets**, but best run appears at `224`; for cost-sensitive deployment, `128` with `resnet18` remains attractive.

5. **The newly added EfficientNet-B0 is the strongest model among all completed runs.**
   Its best configuration is `224 + raw + pretrained`, suggesting that the added preprocessing pipeline is not universally beneficial for every transfer architecture.

6. **VGG11 is competitive but not dominant.**
   With pretraining enabled, `vgg11` reaches 0.8096 accuracy, which is stronger than the custom CNN baselines but still below both `resnet50` and `efficientnet_b0`.

---

## 4. Methods Implemented in This Repository (with Code Links)

### 4.1 Data preprocessing and transforms

- Medical-like optional preprocessing pipeline: grayscale, Gaussian denoise, edge extraction, threshold-mask segmentation, 3-channel merge in [src/brain_tumor_exp/data.py](src/brain_tumor_exp/data.py#L13-L30).
- Train/eval transform construction (resize, augmentation, normalization) in [src/brain_tumor_exp/data.py](src/brain_tumor_exp/data.py#L33-L56).

### 4.2 CNN model implementations

- `SimpleCNN` in [src/brain_tumor_exp/models.py](src/brain_tumor_exp/models.py#L13-L46).
- `SimpleCNNDeeper` (8 conv layers) in [src/brain_tumor_exp/models.py](src/brain_tumor_exp/models.py#L48-L93).
- `resnet18`/`resnet50`/`vgg11`/`efficientnet_b0` model factory in [src/brain_tumor_exp/models.py](src/brain_tumor_exp/models.py#L97-L141).

### 4.3 Experiment grid and ablation logic

- Grid expansion over models, image sizes, preprocessing, pretraining in [src/brain_tumor_exp/runner.py](src/brain_tumor_exp/runner.py#L13-L34).
- Rule that disables pretrained mode for custom CNNs in [src/brain_tumor_exp/runner.py](src/brain_tumor_exp/runner.py#L24-L25).

### 4.4 Training, evaluation, multi-GPU

- Training loop per experiment in [src/brain_tumor_exp/train_eval.py](src/brain_tumor_exp/train_eval.py#L17-L92).
- DataParallel multi-GPU wrapping and checkpoint handling in [src/brain_tumor_exp/train_eval.py](src/brain_tumor_exp/train_eval.py#L38-L39) and [src/brain_tumor_exp/train_eval.py](src/brain_tumor_exp/train_eval.py#L88-L88).
- Evaluation pipeline in [src/brain_tumor_exp/train_eval.py](src/brain_tumor_exp/train_eval.py#L96-L117).

### 4.5 Metrics

- Macro Accuracy/Precision/Recall/F1 + multiclass OVR AUC computation in [src/brain_tumor_exp/utils.py](src/brain_tumor_exp/utils.py#L26-L66).
- Binary AUC helper (used per class) in [src/brain_tumor_exp/utils.py](src/brain_tumor_exp/utils.py#L69-L79).

### 4.6 CLI controls

- CLI arguments for custom grid control (`--model-names`, `--image-sizes`, `--preprocessing-flags`, `--pretrained-flags`, `--main-preset`) in [run_experiments.py](run_experiments.py#L51-L75).

---

## 5. Final Statement

After extending the study with `vgg11` and `efficientnet_b0`, the best overall result is achieved by `efficientnet_b0` with ImageNet pretraining, image size `224`, and no additional preprocessing (accuracy 0.8401, ROC-AUC 0.9637). Among the previously tested models, `resnet50` remains the strongest, while `resnet18` still offers a strong efficiency-performance trade-off.

---

## 6. Reproducibility

### 6.1 Create a fresh Conda environment

```bash
conda create -n brain_tumor_exp python=3.9 -y
conda activate brain_tumor_exp
```

### 6.2 Install dependencies

From project root:

```bash
pip install -r requirements.txt
```

### 6.3 Unzip dataset

From project root:

```bash
unzip dataset.zip
```

### 6.4 Run experiment scripts

Hyperparameter search block:

```bash
bash scripts/r18_imgsz128_pretr_preproc_gridsrch.sh
```

Full experiment block:

```bash
bash scripts/4nets_2imgszs_wwo-pretr_wwo-preproc.sh
```

Added VGG/EfficientNet block:

```bash
bash scripts/2nets_2imgszs_wwo-pretr_wwo-preproc.sh
```

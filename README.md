# Seasonal Color Classifier (timm Models Edition)

This project implements a 12-class seasonal color analysis classifier specialized for face images, using state-of-the-art models via the timm library.

## Project Structure

- **data/**: Data loading and augmentation modules.
- **models/**: Model architecture definitions (using timm).
- **training/**: Training loop, loss functions, and checkpointing (with optional fine-tuning).
- **evaluation/**: Evaluation metrics and analysis.
- **visualization/**: Plotting functions for training history, confusion matrices, etc.
- **utils/**: Helper functions.
- **config.py**: Configuration parameters (updated hyperparameters, augmentation settings, etc.).
- **main.py**: Main script for training, evaluation, and optional hyperparameter optimization.
- **requirements.txt**: Project dependencies (includes timm).
- **viz/**: Directory for generated plots.
- **checkpoints/**: Directory for model checkpoints.

## Supported Models

The following timm model keys are supported (if you pass a partial key, it may be automatically completed):
- hgnetv2_b5.ssld_stage2_ft_in1k
- vit_base_patch16_clip_224.openai_ft_in12k_in1k (pass "vit_base_patch16_clip_224" to auto-complete)
- tf_efficientnetv2_l.in21k_ft_in1k
- hgnetv2_b5.ssld_stage1_in22k_in1k
- hgnet_base.ssld_in1k
- coatnet_2_rw_224.sw_in12k_ft_in1k
- convformer_m36.sail_in22k_ft_in1k
- maxvit_base_tf_512.in1k
- tf_efficientnetv2_xl.in21k_ft_in1k
- convnextv2_huge.fcmae_ft_in1k
- vit_base_patch8_224.augreg2_in21k_ft_in1k
- vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k

The model is instantiated via timm with pretrained weights (if enabled) and its classifier head is adjusted to output 12 classes.

## Usage

1. Install dependencies:
   ```
   pip install -r SeasonalColourClassification/requirements.txt
   ```
2. Prepare your dataset in `dataset/images/train` and `dataset/images/test` (or update paths in config.py).
3. Set your `NEPTUNE_API_TOKEN` as an environment variable.
4. Run training (with optional HPO):
   ```
   python -m SeasonalColourClassification.main --model tf_efficientnetv2_l.in21k_ft_in1k --epochs 250
   ```
   or with HPO:
   ```
   python -m SeasonalColourClassification.main --model vit_base_patch16_clip_224 --epochs 250 --hpo
   ```
5. Monitor your Neptune dashboard for training parameters, plots, and checkpoints.


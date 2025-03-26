#!/usr/bin/env python
"""Main script for seasonal color classification with timm-based models.

Logs training parameters to Neptune for every run.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import neptune
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly.io as pio

from SeasonalColourClassification.config import (
    TRAIN_DIR, TEST_DIR, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, IMG_SIZE as DEFAULT_IMG_SIZE, NUM_CLASSES,
    FINE_TUNE, INITIAL_EPOCHS, FINE_TUNE_LR, HPO_RANGES, HPO_TRIALS, HPO_MAX_EPOCHS,
    AUGMENTATION, DEVICE
)
from SeasonalColourClassification.data.data_loader import get_data_loaders
from SeasonalColourClassification.models.base_model import ModelFactory
from SeasonalColourClassification.training.trainer import Trainer
from SeasonalColourClassification.data.data_augmentation import ColorAugmentation
from SeasonalColourClassification.visualization.visualize import (
    plot_training_history, plot_learning_rate, plot_class_distribution, plot_confusion_matrix
)

def plot_augmentation_example(train_dataset, save_path):
    original, _ = train_dataset[0]
    if isinstance(original, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image
        original = to_pil_image(original)
    aug_transform = ColorAugmentation.get_train_transforms(DEFAULT_IMG_SIZE)
    augmented = aug_transform(original)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    def denorm(tensor):
        tensor = tensor.clone().detach()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor.clamp(0,1)
    import numpy as np
    orig_tensor = torch.tensor(np.array(original)).permute(2,0,1).float() / 255.0
    orig_img = denorm(orig_tensor)
    aug_img = denorm(augmented)
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(orig_img.permute(1,2,0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(aug_img.permute(1,2,0).cpu().numpy())
    axes[1].set_title("Augmented")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def compute_cm(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return confusion_matrix(all_labels, all_preds)

def main():
    parser = argparse.ArgumentParser(description="Seasonal Color Classifier with timm models")
    parser.add_argument("--model", type=str, default="tf_efficientnetv2_l.in21k_ft_in1k", help="Model key to use")
    parser.add_argument("--epochs", type=int, default=250, help="Total epochs (fine-tuning after INITIAL_EPOCHS if enabled)")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter optimization before final training")
    args = parser.parse_args()

    model_img_sizes = {
        "hgnetv2_b5.ssld_stage2_ft_in1k": 288,
        "vit_base_patch16_clip_224.openai_ft_in12k_in1k": 224,
        "tf_efficientnetv2_l.in21k_ft_in1k": 384,
        "hgnetv2_b5.ssld_stage1_in22k_in1k": 288,
        "hgnet_base.ssld_in1k": 224,
        "coatnet_2_rw_224.sw_in12k_ft_in1k": 224,
        "convformer_m36.sail_in22k_ft_in1k": 224,
        "maxvit_base_tf_512.in1k": 512,
        "tf_efficientnetv2_xl.in21k_ft_in1k": 384,
        "convnextv2_huge.fcmae_ft_in1k": 288,
        "vit_base_patch8_224.augreg2_in21k_ft_in1k": 224,
        "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k": 256,
    }
    img_size = model_img_sizes.get(args.model, DEFAULT_IMG_SIZE)

    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    if api_token is None:
        raise ValueError("NEPTUNE_API_TOKEN environment variable not set!")

    local_viz_dir = "viz/hpo_temp"
    os.makedirs(local_viz_dir, exist_ok=True)

    train_loader_example, test_loader_example = get_data_loaders(TRAIN_DIR, TEST_DIR, img_size=img_size)
    train_dataset_example = train_loader_example.dataset
    aug_example_path = os.path.join(local_viz_dir, "augmentation_example.png")
    plot_augmentation_example(train_dataset_example, aug_example_path)

    train_labels = [label for _, label in train_dataset_example.samples]
    test_labels = [label for _, label in test_loader_example.dataset.samples]
    class_balance_path = os.path.join(local_viz_dir, "class_balance.png")
    plot_class_distribution(train_labels, test_labels, train_dataset_example.classes, save_path=class_balance_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_hparams = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "fine_tune_lr": FINE_TUNE_LR,
        "batch_size": BATCH_SIZE
    }

    def objective(trial):
        trial_run = neptune.init_run(
            project="happyproject235/fashion-color-classification",
            api_token=api_token,
            tags=["Armocromia-Dataset", "timm-model", f"{args.model}", "HPO"]
        )
        lr = trial.suggest_float("learning_rate", HPO_RANGES["learning_rate"][0], HPO_RANGES["learning_rate"][1], log=True)
        wd = trial.suggest_float("weight_decay", HPO_RANGES["weight_decay"][0], HPO_RANGES["weight_decay"][1], log=False)
        bsz = trial.suggest_categorical("batch_size", HPO_RANGES["batch_size_options"])
        if FINE_TUNE:
            ftlr = trial.suggest_float("fine_tune_lr", HPO_RANGES["fine_tune_lr"][0], HPO_RANGES["fine_tune_lr"][1], log=True)
        else:
            ftlr = FINE_TUNE_LR

        trial_run["trial/params"] = {
            "learning_rate": lr,
            "weight_decay": wd,
            "fine_tune_lr": ftlr,
            "batch_size": bsz,
            "fine_tune_enabled": FINE_TUNE
        }

        trial_train_loader, trial_test_loader = get_data_loaders(TRAIN_DIR, TEST_DIR, img_size=img_size, batch_size=bsz)
        trial_model = ModelFactory.get_model(args.model)
        ckpt_dir_trial = os.path.join("checkpoints", f"trial_{trial.number}")
        os.makedirs(ckpt_dir_trial, exist_ok=True)

        trial_trainer = Trainer(
            trial_model,
            trial_train_loader,
            trial_test_loader,
            neptune_run=trial_run,
            save_dir=ckpt_dir_trial,
            custom_learning_rate=lr,
            custom_weight_decay=wd,
            custom_fine_tune_lr=ftlr
        )
        trial_trainer.train(num_epochs=HPO_MAX_EPOCHS)
        best_loss = trial_trainer.best_val_loss
        trial_run["trial/best_val_loss"] = best_loss
        trial_run.stop()
        return best_loss

    if args.hpo:
        print("Running hyperparameter optimization (each trial has its own Neptune run)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=HPO_TRIALS)
        fig_hist = plot_optimization_history(study)
        hist_path = os.path.join(local_viz_dir, "hpo_history.png")
        pio.write_image(fig_hist, hist_path)
        if len(study.trials) > 1:
            fig_importance = plot_param_importances(study)
            importance_path = os.path.join(local_viz_dir, "hpo_importances.png")
            pio.write_image(fig_importance, importance_path)
        best_params = study.best_params
        best_hparams["learning_rate"] = best_params["learning_rate"]
        best_hparams["weight_decay"] = best_params["weight_decay"]
        if "fine_tune_lr" in best_params:
            best_hparams["fine_tune_lr"] = best_params["fine_tune_lr"]
        best_hparams["batch_size"] = best_params["batch_size"]
        print("Best hyperparams found by HPO:", best_hparams)

    final_run = neptune.init_run(
        project="happyproject235/fashion-color-classification",
        api_token=api_token,
        tags=["final-training", f"{args.model}", "timm-model"]
    )
    final_run_id = final_run["sys/id"].fetch()
    final_run["training_params"] = {
        "learning_rate": best_hparams["learning_rate"],
        "weight_decay": best_hparams["weight_decay"],
        "fine_tune_lr": best_hparams["fine_tune_lr"],
        "batch_size": best_hparams["batch_size"],
        "epochs": args.epochs,
        "fine_tune_enabled": FINE_TUNE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "img_size": img_size,
        "augmentation": AUGMENTATION,
        "device": str(DEVICE)
    }
    final_run["training_params/notes"] = "Hyperparameters for final training run"

    final_viz_dir = os.path.join("viz", f"final_{final_run_id}")
    final_ckpt_dir = os.path.join("checkpoints", f"final_{final_run_id}")
    os.makedirs(final_viz_dir, exist_ok=True)
    os.makedirs(final_ckpt_dir, exist_ok=True)

    final_train_loader, final_test_loader = get_data_loaders(TRAIN_DIR, TEST_DIR, img_size=img_size, batch_size=best_hparams["batch_size"])
    model = ModelFactory.get_model(args.model)

    trainer = Trainer(
        model,
        final_train_loader,
        final_test_loader,
        save_dir=final_ckpt_dir,
        neptune_run=final_run,
        custom_learning_rate=best_hparams["learning_rate"],
        custom_weight_decay=best_hparams["weight_decay"],
        custom_fine_tune_lr=best_hparams["fine_tune_lr"]
    )
    history = trainer.train(num_epochs=args.epochs)

    training_history_path = os.path.join(final_viz_dir, "training_history.png")
    learning_rate_path = os.path.join(final_viz_dir, "learning_rate.png")
    plot_training_history(history, save_path=training_history_path)
    plot_learning_rate(history, save_path=learning_rate_path)
    final_run["plots/training_history"].upload(training_history_path)
    final_run["plots/learning_rate"].upload(learning_rate_path)

    train_cm = compute_cm(final_train_loader, model, device=device)
    test_cm = compute_cm(final_test_loader, model, device=device)
    train_cm_path = os.path.join(final_viz_dir, "train_confusion_matrix.png")
    test_cm_path = os.path.join(final_viz_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(train_cm, final_train_loader.dataset.classes, title="Train Confusion Matrix", save_path=train_cm_path)
    plot_confusion_matrix(test_cm, final_test_loader.dataset.classes, title="Test Confusion Matrix", save_path=test_cm_path)
    final_run["plots/train_confusion_matrix"].upload(train_cm_path)
    final_run["plots/test_confusion_matrix"].upload(test_cm_path)

    best_ckpt_path = os.path.join(final_ckpt_dir, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        final_run["model/best_checkpoint"].upload(best_ckpt_path)
    if len(history["val_acc"]) > 0:
        final_run["eval/final_accuracy"] = history["val_acc"][-1]
    final_run.stop()

if __name__ == "__main__":
    main()

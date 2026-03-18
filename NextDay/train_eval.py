# coding=utf-8
"""Standalone training and evaluation pipeline for Next Day Wildfire Spread.

This script trains a CNN autoencoder on TFRecord data and evaluates it using
AUC, Precision, Recall, and F1 metrics. It supports multiple experiments
through command-line flags.

Usage:
  python train_eval.py --experiment baseline
  python train_eval.py --experiment ablation --ablate_group weather
  python train_eval.py --experiment hparam --lr 1e-3 --dropout 0.2 --pos_weight 2.0
  python train_eval.py --experiment optimized
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------------------------------------------------------------------
# Path setup: add the Physics_Wildfire root to sys.path so relative
# imports inside the package work when this script is run directly.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from Physics_Wildfire import constants, dataset as ds_module
from Physics_Wildfire.models import cnn_autoencoder_model, metrics as custom_metrics, losses as custom_losses

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
ARCHIVE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'archive')
TRAIN_PATTERN = os.path.join(ARCHIVE_DIR, 'next_day_wildfire_spread_train_*.tfrecord')
EVAL_PATTERN  = os.path.join(ARCHIVE_DIR, 'next_day_wildfire_spread_eval_*.tfrecord')
TEST_PATTERN  = os.path.join(ARCHIVE_DIR, 'next_day_wildfire_spread_test_*.tfrecord')

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'experiment_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature groupings for ablation study
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    'weather':    ['pdsi', 'pr', 'sph', 'th', 'tmmn', 'tmmx', 'vs', 'erc'],
    'topography': ['elevation'],
    'vegetation': ['NDVI'],
    'population': ['population'],
    'prev_fire':  ['PrevFireMask'],
}

ALL_INPUT_FEATURES = list(constants.INPUT_FEATURES)  # 12 features


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
ENCODER_LAYERS = (32, 64, 128)
ENCODER_POOLS  = (1,  2,   2)
DECODER_LAYERS = (128, 64, 32)
DECODER_POOLS  = (2,   2,  2)


def build_model(num_in_channels, dropout=0.0, batch_norm='none',
                l1_reg=0.0, l2_reg=0.0):
  """Build and return a compiled CNN autoencoder Keras model."""
  inputs = tf.keras.Input(shape=(64, 64, num_in_channels), name='input')
  outputs = cnn_autoencoder_model.create_model(
      input_tensor=inputs,
      num_out_channels=1,
      encoder_layers=ENCODER_LAYERS,
      decoder_layers=DECODER_LAYERS,
      encoder_pools=ENCODER_POOLS,
      decoder_pools=DECODER_POOLS,
      dropout=dropout,
      batch_norm=batch_norm,
      l1_regularization=l1_reg,
      l2_regularization=l2_reg,
  )
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model


def compile_model(model, lr=1e-3, pos_weight=1.0, clipnorm=1.0):
  """Compile the model with optimizer, loss, and metrics."""
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
  loss_fn = custom_losses.weighted_cross_entropy_with_logits_with_masked_class(
      pos_weight=pos_weight)
  model.compile(
      optimizer=optimizer,
      loss=loss_fn,
      metrics=[
          custom_metrics.AUCWithMaskedClass(
              with_logits=True, name='auc', curve='ROC'),
          custom_metrics.PrecisionWithMaskedClass(
              with_logits=True, name='precision'),
          custom_metrics.RecallWithMaskedClass(
              with_logits=True, name='recall'),
      ],
  )
  return model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_datasets(input_features, batch_size=32, shuffle_buffer=500):
  """Return (train_ds, eval_ds, test_ds) tf.data.Dataset objects."""
  train_ds = ds_module.make_dataset_from_config(
      file_pattern=TRAIN_PATTERN,
      batch_size=batch_size,
      input_features=tuple(input_features),
      shuffle=True,
      repeat=True,
      random_flip=True,
      random_rotate=True,
      shuffle_buffer_size=shuffle_buffer,
  )
  eval_ds = ds_module.make_dataset_from_config(
      file_pattern=EVAL_PATTERN,
      batch_size=batch_size,
      input_features=tuple(input_features),
      shuffle=False,
      repeat=False,
      random_flip=False,
      random_rotate=False,
  )
  test_ds = ds_module.make_dataset_from_config(
      file_pattern=TEST_PATTERN,
      batch_size=batch_size,
      input_features=tuple(input_features),
      shuffle=False,
      repeat=False,
      random_flip=False,
      random_rotate=False,
  )
  return train_ds, eval_ds, test_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_evaluate(
    experiment_name,
    input_features=None,
    epochs=20,
    steps_per_epoch=200,
    validation_steps=50,
    batch_size=32,
    lr=1e-3,
    dropout=0.0,
    pos_weight=1.0,
    batch_norm='none',
    l1_reg=0.0,
    l2_reg=0.0,
):
  """Train the model, evaluate on val and test sets, and save results."""
  if input_features is None:
    input_features = ALL_INPUT_FEATURES
  num_channels = len(input_features)

  logger.info('=== Experiment: %s ===', experiment_name)
  logger.info('Features (%d): %s', num_channels, input_features)
  logger.info('Hyperparams: epochs=%d, lr=%.4g, dropout=%.2f, pos_weight=%.1f, '
              'batch_size=%d, batch_norm=%s', epochs, lr, dropout, pos_weight,
              batch_size, batch_norm)

  exp_dir = os.path.join(RESULTS_DIR, experiment_name)
  os.makedirs(exp_dir, exist_ok=True)

  # Build and compile model
  model = build_model(num_channels, dropout=dropout, batch_norm=batch_norm,
                      l1_reg=l1_reg, l2_reg=l2_reg)
  model = compile_model(model, lr=lr, pos_weight=pos_weight)
  model.summary(print_fn=logger.info)

  # Datasets
  train_ds, eval_ds, test_ds = get_datasets(
      input_features, batch_size=batch_size)

  # Callbacks
  callbacks = [
      tf.keras.callbacks.EarlyStopping(
          monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-6,
          verbose=1),
      tf.keras.callbacks.CSVLogger(
          os.path.join(exp_dir, 'training_log.csv')),
  ]

  # Train
  t0 = time.time()
  history = model.fit(
      train_ds,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_ds,
      validation_steps=validation_steps,
      callbacks=callbacks,
      verbose=1,
  )
  elapsed = time.time() - t0
  logger.info('Training complete in %.1f seconds.', elapsed)

  # Save weights
  weights_path = os.path.join(exp_dir, 'model_weights.weights.h5')
  model.save_weights(weights_path)
  logger.info('Weights saved to %s', weights_path)

  # Evaluate on test set
  logger.info('Evaluating on test set...')
  test_metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
  logger.info('Test metrics: %s', test_metrics)

  # Compute F1 from precision & recall
  prec = test_metrics.get('precision', 0.0)
  rec  = test_metrics.get('recall', 0.0)
  f1   = 2 * prec * rec / (prec + rec + 1e-8)
  test_metrics['f1'] = float(f1)
  logger.info('Test F1: %.4f', f1)

  # Save all results
  results = {
      'experiment': experiment_name,
      'input_features': input_features,
      'hyperparams': {
          'epochs': epochs, 'lr': lr, 'dropout': dropout,
          'pos_weight': pos_weight, 'batch_size': batch_size,
          'batch_norm': batch_norm,
      },
      'history': {k: [float(v) for v in vs]
                  for k, vs in history.history.items()},
      'test_metrics': {k: float(v) for k, v in test_metrics.items()},
      'training_time_seconds': elapsed,
  }
  with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

  # Plot training curves
  _plot_training_curves(history.history, exp_dir, experiment_name)

  return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_training_curves(history, out_dir, title):
  """Plot and save training/validation loss and AUC curves."""
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  fig.suptitle(f'Training Curves — {title}', fontsize=13)

  # Loss
  ax = axes[0]
  ax.plot(history['loss'], label='Train Loss')
  if 'val_loss' in history:
    ax.plot(history['val_loss'], label='Val Loss')
  ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
  ax.set_title('Loss'); ax.legend(); ax.grid(True, alpha=0.3)

  # AUC
  ax = axes[1]
  ax.plot(history['auc'], label='Train AUC')
  if 'val_auc' in history:
    ax.plot(history['val_auc'], label='Val AUC')
  ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
  ax.set_title('AUC (ROC)'); ax.legend(); ax.grid(True, alpha=0.3)

  plt.tight_layout()
  path = os.path.join(out_dir, 'training_curves.png')
  plt.savefig(path, dpi=150, bbox_inches='tight')
  plt.close()
  logger.info('Training curves saved: %s', path)


def plot_experiment_comparison(all_results, out_dir):
  """Bar chart comparing test AUC, Precision, Recall, F1 across experiments."""
  names = [r['experiment'] for r in all_results]
  metrics_to_plot = ['loss', 'auc', 'precision', 'recall', 'f1']
  display_labels  = ['Loss', 'AUC', 'Precision', 'Recall', 'F1']
  colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

  n = len(metrics_to_plot)
  fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
  fig.suptitle('Experiment Comparison — Test Set', fontsize=14)

  for ax, metric, label, color in zip(axes, metrics_to_plot, display_labels, colors):
    values = [r['test_metrics'].get(metric, 0.0) for r in all_results]
    bars = ax.bar(names, values, color=color, alpha=0.85, edgecolor='white')
    ax.set_title(label); ax.set_ylim(0, max(values) * 1.2 + 0.01)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
              f'{val:.3f}', ha='center', va='bottom', fontsize=8)

  plt.tight_layout()
  path = os.path.join(out_dir, 'experiment_comparison.png')
  plt.savefig(path, dpi=150, bbox_inches='tight')
  plt.close()
  logger.info('Comparison chart saved: %s', path)


def plot_ablation_results(ablation_results, baseline_result, out_dir):
  """Bar chart showing impact of removing each feature group."""
  # baseline is the 'all features' result
  baseline_auc = baseline_result['test_metrics']['auc']
  names  = [r['experiment'].replace('ablation_', '') for r in ablation_results]
  aucs   = [r['test_metrics']['auc'] for r in ablation_results]
  deltas = [baseline_auc - a for a in aucs]

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle('Feature Ablation Study', fontsize=13)

  ax = axes[0]
  bars = ax.bar(names, aucs, color='#3498db', alpha=0.85, edgecolor='white')
  ax.axhline(baseline_auc, color='red', linestyle='--', label=f'Baseline AUC={baseline_auc:.3f}')
  ax.set_title('AUC with feature group removed')
  ax.set_ylabel('Test AUC'); ax.legend(); ax.grid(axis='y', alpha=0.3)
  ax.tick_params(axis='x', rotation=20)
  for bar, v in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9)

  ax = axes[1]
  colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in deltas]
  ax.bar(names, deltas, color=colors, alpha=0.85, edgecolor='white')
  ax.axhline(0, color='black', linewidth=0.8)
  ax.set_title('AUC drop vs. baseline (higher = more important group)')
  ax.set_ylabel('ΔAUC'); ax.grid(axis='y', alpha=0.3)
  ax.tick_params(axis='x', rotation=20)

  plt.tight_layout()
  path = os.path.join(out_dir, 'ablation_results.png')
  plt.savefig(path, dpi=150, bbox_inches='tight')
  plt.close()
  logger.info('Ablation chart saved: %s', path)


def plot_hparam_results(hparam_results, out_dir):
  """Heatmap-style summary of hyperparameter search results."""
  names = [r['experiment'] for r in hparam_results]
  aucs  = [r['test_metrics']['auc'] for r in hparam_results]
  f1s   = [r['test_metrics']['f1'] for r in hparam_results]

  x = range(len(names))
  fig, ax = plt.subplots(figsize=(max(8, len(names)*1.2), 5))
  ax.bar([i - 0.2 for i in x], aucs, width=0.4,
         label='AUC', color='#3498db', alpha=0.85)
  ax.bar([i + 0.2 for i in x], f1s,  width=0.4,
         label='F1',  color='#e67e22', alpha=0.85)
  ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=30, ha='right')
  ax.set_ylabel('Score'); ax.set_title('Hyperparameter Search Results')
  ax.legend(); ax.grid(axis='y', alpha=0.3)
  plt.tight_layout()
  path = os.path.join(out_dir, 'hparam_results.png')
  plt.savefig(path, dpi=150, bbox_inches='tight')
  plt.close()
  logger.info('Hparam chart saved: %s', path)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test():
  """Load one batch and verify shapes."""
  logger.info('Running smoke test...')
  train_ds, eval_ds, test_ds = get_datasets(ALL_INPUT_FEATURES, batch_size=4)
  for x, y in train_ds.take(1):
    logger.info('Input shape:  %s', x.shape)
    logger.info('Output shape: %s', y.shape)
    assert x.shape == (4, 64, 64, 12), f'Unexpected input shape: {x.shape}'
    assert y.shape == (4, 64, 64, 1),  f'Unexpected output shape: {y.shape}'
  logger.info('Smoke test PASSED.')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_args():
  p = argparse.ArgumentParser(description='Wildfire spread prediction training.')
  p.add_argument('--experiment', default='baseline',
                 choices=['smoke', 'baseline', 'ablation', 'hparam', 'optimized', 'all'],
                 help='Which experiment to run.')
  p.add_argument('--ablate_group', default='weather',
                 choices=list(FEATURE_GROUPS.keys()),
                 help='Feature group to ablate (used with --experiment ablation).')
  p.add_argument('--epochs',        type=int,   default=20)
  p.add_argument('--steps_per_epoch', type=int, default=200)
  p.add_argument('--val_steps',     type=int,   default=50)
  p.add_argument('--batch_size',    type=int,   default=32)
  p.add_argument('--lr',            type=float, default=1e-3)
  p.add_argument('--dropout',       type=float, default=0.0)
  p.add_argument('--pos_weight',    type=float, default=2.0)
  p.add_argument('--batch_norm',    default='none',
                 choices=['none', 'some', 'all'])
  return p.parse_args()


def run_all_experiments(args):
  """Run all four experiments sequentially."""
  all_results = []

  # ---- Exp 1: Baseline ----
  logger.info('\n\n######  Experiment 1: Baseline  ######')
  baseline = train_and_evaluate(
      'baseline',
      input_features=ALL_INPUT_FEATURES,
      epochs=args.epochs,
      steps_per_epoch=args.steps_per_epoch,
      validation_steps=args.val_steps,
      batch_size=args.batch_size,
      lr=1e-3, dropout=0.0, pos_weight=2.0, batch_norm='none',
  )
  all_results.append(baseline)

  # ---- Exp 2: Feature Ablation ----
  logger.info('\n\n######  Experiment 2: Feature Ablation  ######')
  ablation_results = []
  for group_name, group_feats in FEATURE_GROUPS.items():
    remaining = [f for f in ALL_INPUT_FEATURES if f not in group_feats]
    r = train_and_evaluate(
        f'ablation_{group_name}',
        input_features=remaining,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=1e-3, dropout=0.0, pos_weight=2.0, batch_norm='none',
    )
    ablation_results.append(r)
    all_results.append(r)

  plot_ablation_results(ablation_results, baseline, RESULTS_DIR)

  # ---- Exp 3: Hyperparameter Search ----
  logger.info('\n\n######  Experiment 3: Hyperparameter Search  ######')
  hparam_grid = [
      dict(lr=1e-2, dropout=0.0, pos_weight=2.0, batch_norm='none'),
      dict(lr=1e-3, dropout=0.2, pos_weight=2.0, batch_norm='none'),
      dict(lr=1e-3, dropout=0.0, pos_weight=5.0, batch_norm='none'),
      dict(lr=1e-3, dropout=0.2, pos_weight=5.0, batch_norm='some'),
      dict(lr=1e-4, dropout=0.3, pos_weight=2.0, batch_norm='some'),
  ]
  hparam_results = []
  for i, hp in enumerate(hparam_grid):
    r = train_and_evaluate(
        f'hparam_{i}',
        input_features=ALL_INPUT_FEATURES,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        **hp,
    )
    hparam_results.append(r)
    all_results.append(r)
  plot_hparam_results(hparam_results, RESULTS_DIR)

  # Find best hyperparams
  best_hp_result = max(hparam_results, key=lambda r: r['test_metrics']['auc'])
  best_hp = best_hp_result['hyperparams']
  logger.info('Best hparam config: %s', best_hp)

  # ---- Exp 4: Optimized Model ----
  logger.info('\n\n######  Experiment 4: Optimized Model  ######')
  optimized = train_and_evaluate(
      'optimized',
      input_features=ALL_INPUT_FEATURES,
      epochs=max(args.epochs * 2, 40),
      steps_per_epoch=args.steps_per_epoch,
      validation_steps=args.val_steps,
      batch_size=args.batch_size,
      lr=best_hp['lr'],
      dropout=best_hp['dropout'],
      pos_weight=best_hp['pos_weight'],
      batch_norm=best_hp['batch_norm'],
  )
  all_results.append(optimized)

  # Overall comparison chart
  plot_experiment_comparison(
      [baseline, optimized] + hparam_results[:3],
      RESULTS_DIR,
  )

  # Save combined summary
  summary = {exp['experiment']: exp['test_metrics'] for exp in all_results}
  with open(os.path.join(RESULTS_DIR, 'all_experiment_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  logger.info('All experiments done. Summary saved.')
  return all_results


if __name__ == '__main__':
  args = parse_args()

  if args.experiment == 'smoke':
    smoke_test()

  elif args.experiment == 'baseline':
    train_and_evaluate(
        'baseline',
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, pos_weight=args.pos_weight,
        batch_norm=args.batch_norm,
    )

  elif args.experiment == 'ablation':
    group_feats = FEATURE_GROUPS[args.ablate_group]
    remaining = [f for f in ALL_INPUT_FEATURES if f not in group_feats]
    train_and_evaluate(
        f'ablation_{args.ablate_group}',
        input_features=remaining,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, pos_weight=args.pos_weight,
    )

  elif args.experiment == 'hparam':
    train_and_evaluate(
        f'hparam_lr{args.lr}_do{args.dropout}_pw{args.pos_weight}',
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, pos_weight=args.pos_weight,
        batch_norm=args.batch_norm,
    )

  elif args.experiment == 'optimized':
    train_and_evaluate(
        'optimized',
        epochs=max(args.epochs * 2, 40),
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        batch_size=args.batch_size,
        lr=args.lr, dropout=args.dropout, pos_weight=args.pos_weight,
        batch_norm=args.batch_norm,
    )

  elif args.experiment == 'all':
    run_all_experiments(args)

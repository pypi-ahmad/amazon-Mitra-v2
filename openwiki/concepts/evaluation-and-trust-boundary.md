---
type: Evaluation Concept
title: Evaluation and Trust Boundary
description: Defines the known/hidden-row evaluation used by Glass Box, how MITRA and its reference baseline are compared, and what those scores cannot establish.
tags: [evaluation, data-quality, metrics, trust-boundary]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:16:42.939Z
sources:
  - id: openwiki-source-98a331068a0dc759cb5dd224
    resource: repo://app_pages/results.py
  - id: openwiki-source-68b3e60cd84c1d81b154ac7d
    resource: repo://scripts/smoke_classifier.py
  - id: openwiki-source-65741d003485f1a410e67e25
    resource: repo://scripts/smoke_data.py
  - id: openwiki-source-d142615b2faeb5e851e19033
    resource: repo://src/data_catalog.py
  - id: openwiki-source-86ef16584268767232778345
    resource: repo://src/load.py
  - id: openwiki-source-ec840809b2d30144c3aaebd1
    resource: repo://src/mitra_run.py
generated: { by: "codex", at: "2026-09-23T14:16:42.939Z" }
---

# Evaluation and Trust Boundary

Glass Box reports predictive performance on a deterministic hidden subset of the selected table. These measurements describe the current table, target, split, checkpoint, and run settings. They do not certify the input data or support causal conclusions.

## How the hidden split is created

For a catalog table or upload, `build_metadata` records the target, task, split size, random seed, and sorted hidden-row indices. The default catalog policy uses a 10% hidden fraction and seed 42. For classification, the splitter uses stratification only when there are enough examples in every class and enough rows on both sides; otherwise it uses the deterministic unstratified split. Rows with missing target values are excluded from the split indices.

`split_from_metadata` reconstructs copies of known training rows and hidden test rows from those indices. The Data page and training workflow use this separation so the hidden targets remain outside the model-fit data.

## What the runner compares

`run_mitra` accepts separate training and hidden tables, checks that both contain the target and matching feature columns, then fits the task-appropriate released MITRA head. It predicts the hidden rows and computes metrics against their target values. The runner fits one HistGradientBoosting reference model using the same known rows and evaluates it on the same hidden rows.

Regression reports RMSE, MAE, and R²; it adds MAPE only when all observed targets are positive. Classification reports accuracy and F1, adds binary ROC-AUC when probabilities are available, and builds a confusion matrix. The reference model gives a common comparison point; it is not a full AutoML benchmark or a guarantee that model choices are otherwise comparable.

## Interpret results within their scope

The Results page warns that model results do not certify that a table is accurate, representative, fair, causally valid, or trustworthy. A hidden-row score can expose predictive behavior under the chosen split, but it cannot establish those properties of the data or explain why a prediction is correct.

The focused checks verify implementation contracts: the data smoke checks split coverage and source behavior; the classifier and regressor smokes check finite metrics and required run artifacts. Those checks do not substitute for a broader benchmark or independent dataset audit. See [Data Selection and Splitting](../workflows/data-selection-and-splitting.md), [Model Training and Results](../workflows/model-training-and-results.md), and the [Smoke Check Map](../testing/smoke-checks.md).

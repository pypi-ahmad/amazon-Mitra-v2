# Graph Report - amazon-Mitra-v2  (2026-09-23)

## Corpus Check
- 81 files · ~358,024 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 17 file(s) not represented in the graph (top: .csv 7, (none) 6, .example 1)

## Summary
- 478 nodes · 705 edges · 78 communities (35 shown, 43 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.92)
- Token usage: unavailable (the host execution API did not report usage)

## Community Hubs (Navigation)
- Project Documentation
- EDA and Visualization
- MITRA Model Runner
- Dataset Catalog
- Run Sequence Light
- Run Sequence Dark
- Training Pages and Uploads
- Classifier Smoke Checks
- Streamlit Pages
- App Startup and State
- Dataset Smoke Tests
- Workflow Diagram Steps
- Synthetic EDA Gallery
- Dataflow Screenshot 2048 Dark
- Dataflow Screenshot 1440 Dark
- Dataflow Screenshot 1440 Light
- Dataflow Screenshot 2048 Light
- Runtime Dependencies
- GPU Fine Tuning Smoke
- Documentation Smoke Check
- Architecture Screenshot 1440 Dark
- Architecture Screenshot 1440 Light
- Architecture Screenshot 2048 Dark
- Architecture Screenshot 2048 Light
- Lifecycle Legend and Stages
- Workflow Steps 2048 Dark
- Run Sequence Participants 1440 Light
- Run Sequence Participants 2048 Light
- Workflow Steps 1440 Light
- Lifecycle Stages 1440 Dark
- Lifecycle Stages 1440 Light
- Lifecycle Stages 2048 Dark
- Fallback Data Smoke Test
- Windows Launcher Smoke
- Run Sequence Phases
- Workflow Summary 1440 Light
- Workflow Summary 2048 Light
- Shared Source Package
- Windows App Architecture
- Architecture Heading 1440 Dark
- Single Process 1440 Dark
- Architecture Heading 1440 Light
- Single Process 1440 Light
- Architecture Heading 2048 Dark
- Single Process 2048 Dark
- Architecture Heading 2048 Light
- Single Process 2048 Light
- Dataflow Heading 1440 Dark
- Dataflow Heading 1440 Light
- Dataflow Heading 2048 Dark
- Dataflow Heading 2048 Light
- Lifecycle Heading 1440 Dark
- Recovery Notes 1440 Dark
- Lifecycle Outcomes 1440 Dark
- Run Phases 1440 Dark
- Lifecycle Heading 1440 Light
- Recovery Notes 1440 Light
- Lifecycle Outcomes 1440 Light
- Lifecycle Heading 2048 Dark
- Lifecycle Diagram Heading 2048 Dark
- Recovery Notes 2048 Dark
- Lifecycle Outcomes 2048 Dark
- Lifecycle Heading 2048 Light
- Recovery Notes 2048 Light
- Lifecycle Outcomes 2048 Light
- Run Phases 2048 Light
- Sequence Diagram Heading 1440 Dark
- Sequence Diagram Heading 2048 Dark
- Sequence Diagram Heading 2048 Light
- Workflow Diagram Heading 2048 Dark
- Workflow Choose Table
- Workflow Fit MITRA
- Workflow Load Table
- Workflow Predict
- Workflow Show Error
- Workflow Show Results
- Workflow Split Rows
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `load_sample()` - 22 edges
2. `run_mitra()` - 21 edges
3. `split_from_metadata()` - 10 edges
4. `current_data()` - 10 edges
5. `Bundled Dataset Catalog` - 10 edges
6. `main()` - 9 edges
7. `build_upload_metadata()` - 9 edges
8. `calculate_metrics()` - 9 edges
9. `SourceSpec` - 8 edges
10. `build_metadata()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `load_sample()`  [EXTRACTED]
  scripts/smoke_classifier.py → src/load.py
- `main()` --calls--> `run_mitra()`  [EXTRACTED]
  scripts/smoke_classifier.py → src/mitra_run.py
- `main()` --calls--> `load_sample()`  [EXTRACTED]
  scripts/smoke_data.py → src/load.py
- `main()` --calls--> `load_sample()`  [EXTRACTED]
  scripts/smoke_eda.py → src/load.py
- `_run_regression()` --calls--> `load_sample()`  [EXTRACTED]
  scripts/smoke_finetune_gpu.py → src/load.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Glass Box tabular evaluation workflow** — docs_architecture_streamlit_app, docs_architecture_deterministic_split, docs_architecture_mitra_runner [EXTRACTED 1.00]
- **Five primary tutorial datasets** — docs_datasets_california_housing, docs_datasets_predictive_maintenance, docs_datasets_adult_census_income, docs_datasets_german_credit, docs_datasets_wine_quality [EXTRACTED 1.00]
- **Persisted outputs from a successful run** — docs_architecture_mitra_runner, docs_architecture_histgradientboosting_baseline, docs_architecture_run_artifacts [EXTRACTED 1.00]
- **Components inside one native Windows Streamlit process** — docs_diagrams_mitra_architecture_visual_check_1440x900_dark_run_cmd, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_streamlit_app, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_dataset_catalog, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_load_split, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_session_state, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_mitra_runner, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_mitra_v2_heads, docs_diagrams_mitra_architecture_visual_check_1440x900_dark_run_artifacts [EXTRACTED 1.00]
- **Components Inside One Native Windows Streamlit Process** — docs_diagrams_mitra_architecture_visual_check_1440x900_light_run_cmd, docs_diagrams_mitra_architecture_visual_check_1440x900_light_streamlit_app, docs_diagrams_mitra_architecture_visual_check_1440x900_light_dataset_catalog, docs_diagrams_mitra_architecture_visual_check_1440x900_light_load_and_split, docs_diagrams_mitra_architecture_visual_check_1440x900_light_session_state, docs_diagrams_mitra_architecture_visual_check_1440x900_light_mitra_runner, docs_diagrams_mitra_architecture_visual_check_1440x900_light_mitra_v2_heads, docs_diagrams_mitra_architecture_visual_check_1440x900_light_run_artifacts [EXTRACTED 1.00]
- **Components inside one native Windows Streamlit process** — docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_run_cmd, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_streamlit_app, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_dataset_catalog, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_load_split, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_session_state, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_mitra_runner, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_mitra_v2_head, docs_diagrams_mitra_architecture_visual_check_2048x1320_dark_run_artifacts [EXTRACTED 1.00]
- **Components within one native Windows Streamlit process** — docs_diagrams_mitra_architecture_visual_check_2048x1320_light_one_native_windows_streamlit_process, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_run_cmd, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_streamlit_app, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_dataset_catalog, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_load_and_split, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_session_state, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_mitra_runner, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_mitra_v2_heads, docs_diagrams_mitra_architecture_visual_check_2048x1320_light_run_artifacts [EXTRACTED 1.00]
- **Dataset-to-results workflow** — docs_diagrams_mitra_dataflow_visual_check_1440x900_light_hf_datasets, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_table_upload, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_local_fallback, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_normalized_table, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_known_rows, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_hidden_rows, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_mitra_prediction, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_run_artifacts, docs_diagrams_mitra_dataflow_visual_check_1440x900_light_results_page [EXTRACTED 1.00]
- **Dataset-to-results pipeline** — docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_hf_datasets, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_table_upload, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_local_fallback, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_normalized_table, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_known_rows, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_hidden_rows, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_mitra_prediction, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_run_artifacts, docs_diagrams_mitra_dataflow_visual_check_2048x1320_dark_results_page [EXTRACTED 1.00]
- **Dataset sources converge into a normalized table** — docs_diagrams_mitra_dataflow_visual_check_2048x1320_light_hf_datasets, docs_diagrams_mitra_dataflow_visual_check_2048x1320_light_table_upload, docs_diagrams_mitra_dataflow_visual_check_2048x1320_light_local_fallback, docs_diagrams_mitra_dataflow_visual_check_2048x1320_light_normalized_table [EXTRACTED 1.00]
- **Sequential run phase flow** — docs_diagrams_mitra_lifecycle_visual_check_1440x900_dark_idle, docs_diagrams_mitra_lifecycle_visual_check_1440x900_dark_loading, docs_diagrams_mitra_lifecycle_visual_check_1440x900_dark_prepared, docs_diagrams_mitra_lifecycle_visual_check_1440x900_dark_fitting, docs_diagrams_mitra_lifecycle_visual_check_1440x900_dark_persisted [EXTRACTED 1.00]
- **Run Phases** — docs_diagrams_mitra_lifecycle_visual_check_1440x900_light_idle, docs_diagrams_mitra_lifecycle_visual_check_1440x900_light_loading, docs_diagrams_mitra_lifecycle_visual_check_1440x900_light_prepared, docs_diagrams_mitra_lifecycle_visual_check_1440x900_light_fitting, docs_diagrams_mitra_lifecycle_visual_check_1440x900_light_persisted [EXTRACTED 1.00]
- **Run phases form the Glass Box Run Lifecycle** — docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_glass_box_run_lifecycle, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_idle, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_loading, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_prepared, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_fitting, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_dark_persisted [EXTRACTED 1.00]
- **Ordered run phase flow** — docs_diagrams_mitra_lifecycle_visual_check_2048x1320_light_idle, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_light_loading, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_light_prepared, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_light_fitting, docs_diagrams_mitra_lifecycle_visual_check_2048x1320_light_persisted [EXTRACTED 1.00]
- **Glass Box Run Request Flow** — docs_diagrams_mitra_sequence_visual_check_1440x900_dark_user, docs_diagrams_mitra_sequence_visual_check_1440x900_dark_streamlit, docs_diagrams_mitra_sequence_visual_check_1440x900_dark_loader, docs_diagrams_mitra_sequence_visual_check_1440x900_dark_mitra_runner, docs_diagrams_mitra_sequence_visual_check_1440x900_dark_mitra_v2, docs_diagrams_mitra_sequence_visual_check_1440x900_dark_run_directory [EXTRACTED 1.00]
- **Participants in the Glass Box Run Request sequence** — docs_diagrams_mitra_sequence_visual_check_1440x900_light_user, docs_diagrams_mitra_sequence_visual_check_1440x900_light_streamlit, docs_diagrams_mitra_sequence_visual_check_1440x900_light_loader, docs_diagrams_mitra_sequence_visual_check_1440x900_light_mitra_runner, docs_diagrams_mitra_sequence_visual_check_1440x900_light_mitra_v2, docs_diagrams_mitra_sequence_visual_check_1440x900_light_run_directory [EXTRACTED 1.00]
- **Participants in the MITRA run request workflow** — docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_user, docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_streamlit_train_page, docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_loader, docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_mitra_runner, docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_mitra_v2, docs_diagrams_mitra_sequence_visual_check_2048x1320_dark_run_directory [EXTRACTED 1.00]
- **Prepare** — docs_diagrams_mitra_sequence_visual_check_2048x1320_light_user, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_streamlit, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_loader, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_click_run, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_resolve_selected_data, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_known_and_hidden_tables, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_run_mitra_settings [EXTRACTED 1.00]
- **Fit and Predict** — docs_diagrams_mitra_sequence_visual_check_2048x1320_light_mitra_runner, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_mitra_v2, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_run_directory, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_select_classifier_or_regressor, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_fit_autogluon_mitra, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_hidden_row_predictions, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_write_baseline_metrics, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_write_metrics_and_predictions, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_persist_predictor [EXTRACTED 1.00]
- **Persist and Display** — docs_diagrams_mitra_sequence_visual_check_2048x1320_light_mitra_runner, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_streamlit, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_run_directory, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_result_metadata, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_metrics_and_status, docs_diagrams_mitra_sequence_visual_check_2048x1320_light_error_json_and_run_log [EXTRACTED 1.00]
- **Visible training and prediction flow** — docs_diagrams_mitra_workflow_visual_check_1440x900_dark_choose_table, docs_diagrams_mitra_workflow_visual_check_1440x900_dark_load_table, docs_diagrams_mitra_workflow_visual_check_1440x900_dark_split_rows, docs_diagrams_mitra_workflow_visual_check_1440x900_dark_fit_mitra, docs_diagrams_mitra_workflow_visual_check_1440x900_dark_predict, docs_diagrams_mitra_workflow_visual_check_1440x900_dark_show_results [EXTRACTED 1.00]
- **Training Workflow Stages** — docs_diagrams_mitra_workflow_visual_check_1440x900_light_choose_table, docs_diagrams_mitra_workflow_visual_check_1440x900_light_load_table, docs_diagrams_mitra_workflow_visual_check_1440x900_light_split_rows, docs_diagrams_mitra_workflow_visual_check_1440x900_light_fit_mitra, docs_diagrams_mitra_workflow_visual_check_1440x900_light_predict, docs_diagrams_mitra_workflow_visual_check_1440x900_light_show_results [EXTRACTED 1.00]
- **Glass Box Training Workflow Steps** — docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_choose_table, docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_load_table, docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_split_rows, docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_fit_mitra, docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_predict, docs_diagrams_mitra_workflow_visual_check_2048x1320_dark_show_results [EXTRACTED 1.00]
- **Steps in the Glass Box Training Workflow** — docs_diagrams_mitra_workflow_visual_check_2048x1320_light_glass_box_training_workflow, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_choose_table, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_load_table, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_split_rows, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_fit_mitra, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_predict, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_show_results, docs_diagrams_mitra_workflow_visual_check_2048x1320_light_show_error [EXTRACTED 1.00]

## Communities (78 total, 43 thin omitted)

### Community 0 - "Project Documentation"
Cohesion: 0.06
Nodes (43): Contribution Guide, Glass Box Architecture, Deterministic Known and Hidden Split, HistGradientBoosting Baseline, Mitra Runner, Official Regression Head Compatibility Patch, Persisted Run Artifacts, Model Results Do Not Certify Data Trustworthiness (+35 more)

### Community 1 - "EDA and Visualization"
Cohesion: 0.10
Nodes (35): Render the classic EDA workflow for the current table., matplotlib, matplotlib_pyplot, main(), Render and save the required classic EDA figures for the Houses sample., Create the EDA smoke artifacts and record the saved plot count., seaborn, correlation_figure() (+27 more)

### Community 2 - "MITRA Model Runner"
Cohesion: 0.11
Nodes (33): datetime, sklearn_compose, sklearn_ensemble, sklearn_impute, sklearn_metrics, sklearn_pipeline, sklearn_preprocessing, calculate_metrics() (+25 more)

### Community 3 - "Dataset Catalog"
Cohesion: 0.17
Nodes (22): dataclasses, io, Path, Definitions for the five bundled tutorial datasets and their fallbacks., Describe one downloadable or local dataset source. Attributes: kind: Loader…, Describe a normalized tutorial table and its deterministic split policy.…, SampleSpec, SourceSpec (+14 more)

### Community 4 - "Run Sequence Light"
Cohesion: 0.19
Nodes (20): Click Run, Fit and predict phase, Fit AutoGluon MITRA, Glass Box Run Request, Hidden-row predictions, Known and hidden tables, Loader (src/load.py), MITRA runner (src/mitra_run.py) (+12 more)

### Community 5 - "Run Sequence Dark"
Cohesion: 0.15
Nodes (19): Click Run, error.json and run.log, Fit AutoGluon MITRA, Hidden-Row Predictions, Known and Hidden Tables, Loader (src/load.py), Metrics and Status, MITRA Runner (src/mitra_run.py) (+11 more)

### Community 6 - "Training Pages and Uploads"
Cohesion: 0.15
Nodes (15): Render uploads, deterministic split details, and row-level data inspection., Render MITRA execution controls and the run log., Parse an uploaded CSV or Parquet file into nullable pandas dtypes. Args:…, read_upload(), clear_upload(), current_data(), infer_problem(), DataFrame (+7 more)

### Community 7 - "Classifier Smoke Checks"
Cohesion: 0.19
Nodes (13): math, _assert_artifacts(), main(), Smoke-test the released Mitra-v2 classifier on a bounded Machines sample., Run and validate a zero-shot, single-copy classification model smoke., main(), Run the zero-shot regression and classification MITRA smoke pair., Run both released-head smokes in their required order. (+5 more)

### Community 8 - "Streamlit Pages"
Cohesion: 0.21
Nodes (9): Render the Glass Box overview and bundled-table selection page., Render locally generated synthetic-prior illustrations., Render a downloadable standalone Python reproduction script for the current…, Render saved or in-session MITRA metrics, predictions, and comparison views., pathlib, plotly_express, plotly_graph_objects, Central project paths, released checkpoint IDs, and source links. (+1 more)

### Community 9 - "App Startup and State"
Cohesion: 0.19
Nodes (11): Streamlit entry point that initializes shared UI state and page navigation., os, initialize_state(), Populate missing Streamlit session-state defaults without overwriting choices., apply_theme(), Render the shared dark Glass Box theme and navigation., Inject the app-wide dark navy and purple Streamlit CSS theme., Render shared dataset, fine-tune-step, and time-limit controls in the sidebar. (+3 more)

### Community 10 - "Dataset Smoke Tests"
Cohesion: 0.18
Nodes (9): Render the traditional-workflow and MITRA workflow comparison., pandas, main(), Download every primary dataset source and record the catalog smoke result., Exercise catalog loading, caching, normalization, and deterministic splits., Return copies of known training rows and hidden test rows from metadata. Args:…, split_from_metadata(), Small compatibility exports used by the tutorial's training-oriented pages. (+1 more)

### Community 11 - "Workflow Diagram Steps"
Cohesion: 0.26
Nodes (12): Choose table (bundled or upload), Data Preparation, Evaluation Boundary, Fit MITRA (task head), Glass Box Training Workflow, Load table (normalize and cache), Model Execution, Predict (MITRA and baseline) (+4 more)

### Community 12 - "Synthetic EDA Gallery"
Cohesion: 0.21
Nodes (10): importlib, numpy, Import the app and verify its deterministic synthetic-gallery inputs., sklearn_datasets, _frame(), DataFrame, ndarray, Create deterministic synthetic pattern illustrations for the gallery. (+2 more)

### Community 13 - "Dataflow Screenshot 2048 Dark"
Cohesion: 0.24
Nodes (10): Data store, HF datasets, Hidden rows, Known rows, Local fallback, MITRA prediction, Normalized table, Results page (+2 more)

### Community 14 - "Dataflow Screenshot 1440 Dark"
Cohesion: 0.25
Nodes (9): HF datasets, Hidden rows, Known rows, Local fallback, MITRA prediction, Normalized table, Results page, Run artifacts (+1 more)

### Community 15 - "Dataflow Screenshot 1440 Light"
Cohesion: 0.25
Nodes (9): HF datasets, Hidden rows, Known rows, Local fallback, MITRA prediction, Normalized table, Results page, Run artifacts (+1 more)

### Community 16 - "Dataflow Screenshot 2048 Light"
Cohesion: 0.25
Nodes (9): HF datasets, Hidden rows, Known rows, Local fallback, MITRA prediction, Normalized table, Results page, Run artifacts (+1 more)

### Community 17 - "Runtime Dependencies"
Cohesion: 0.22
Nodes (9): AutoGluon Tabular with Mitra Extra, Hugging Face datasets, NumPy, pandas, Plotly, Python Runtime Requirements, scikit-learn, Streamlit (+1 more)

### Community 18 - "GPU Fine Tuning Smoke"
Cohesion: 0.39
Nodes (8): main(), Run CUDA-gated fine-tuning and eight-copy MITRA smoke tests for both heads., Skip on CPU or record successful fine-tuned regression and classification runs., _run(), _run_classification(), _run_regression(), _write_pointer(), torch

### Community 19 - "Documentation Smoke Check"
Cohesion: 0.32
Nodes (7): ast, re, _check_documents_and_links(), _check_public_src_docstrings(), main(), Verify required documentation, repository-relative Markdown links, and public…, Run the static documentation contract without data or model downloads.

### Community 20 - "Architecture Screenshot 1440 Dark"
Cohesion: 0.29
Nodes (8): Dataset catalog — sources + fallbacks, Load + split — normalize, cache, 90/10, MITRA runner — head, fit, baseline, MITRA-v2 heads — classifier or regressor, Run artifacts — run evidence, run.cmd — Windows launcher, Session state — settings, Streamlit app — app.py + pages

### Community 21 - "Architecture Screenshot 1440 Light"
Cohesion: 0.29
Nodes (8): Dataset Catalog, Load and Split, MITRA Runner, MITRA-v2 Heads (Classifier or Regressor), Run Artifacts, run.cmd Windows Launcher, Session State, Streamlit App (app.py and pages)

### Community 22 - "Architecture Screenshot 2048 Dark"
Cohesion: 0.29
Nodes (8): Dataset Catalog, Load and Split, MITRA Runner, MITRA-v2 Head, Run Artifacts, run.cmd Windows Launcher, Session State, Streamlit App

### Community 23 - "Architecture Screenshot 2048 Light"
Cohesion: 0.29
Nodes (8): Dataset Catalog (sources + fallbacks), Load + Split (normalize, cache, 90/10), MITRA Runner (head, fit, baseline), MITRA-v2 Heads (classifier or regressor), Run Artifacts (run evidence), run.cmd Windows Launcher, Session State (settings), Streamlit App (app.py + pages)

### Community 24 - "Lifecycle Legend and Stages"
Cohesion: 0.32
Nodes (8): Active state legend category, Fitting — MITRA plus baseline, Idle — ready for run, Loading — resolve table, Persisted — run artifacts, Prepared — split and validate, Start state legend category, Terminal success legend category

### Community 25 - "Workflow Steps 2048 Dark"
Cohesion: 0.29
Nodes (7): Choose Table, Fit MITRA, Load Table, Predict, Show Error, Show Results, Split Rows

### Community 26 - "Run Sequence Participants 1440 Light"
Cohesion: 0.33
Nodes (6): Loader, MITRA runner, MITRA-v2, Run directory, Streamlit, User

### Community 27 - "Run Sequence Participants 2048 Light"
Cohesion: 0.40
Nodes (6): Loader (src/load.py), MITRA runner (src/mitra_run.py), MITRA-v2 on Hugging Face, Run directory (data/runs/<UTC>), Streamlit train page, User

### Community 28 - "Workflow Steps 1440 Light"
Cohesion: 0.33
Nodes (6): Choose Table, Fit MITRA, Load Table, Predict, Show Results, Split Rows

### Community 29 - "Lifecycle Stages 1440 Dark"
Cohesion: 0.40
Nodes (5): Fitting — MIRA + baseline, Idle — ready for run, Loading — resolve table, Persisted — run artifacts, Prepared — split + validate

### Community 30 - "Lifecycle Stages 1440 Light"
Cohesion: 0.40
Nodes (5): Fitting, Idle, Loading, Persisted, Prepared

### Community 31 - "Lifecycle Stages 2048 Dark"
Cohesion: 0.40
Nodes (5): Fitting (MIRA + baseline), Idle, Loading, Persisted, Prepared

### Community 32 - "Fallback Data Smoke Test"
Cohesion: 0.40
Nodes (4): importlib_util, main(), Check installed imports and the five bundled local fallback CSV files., Run dependency and fallback-table assertions.

### Community 33 - "Windows Launcher Smoke"
Cohesion: 0.40
Nodes (4): json, main(), Statically verify the Windows launcher contract without starting Streamlit., Assert launcher setup, token guard, port cleanup, and app command details.

### Community 34 - "Run Sequence Phases"
Cohesion: 0.67
Nodes (3): Fit and predict phase, Persist and display phase, Prepare phase

## Ambiguous Edges - Review These
- `Persist and display phase (partially visible)` → `Persist predictor`  [AMBIGUOUS]
  docs/diagrams/mitra-sequence.visual-check.1440x900.light.png · relation: conceptually_related_to

## Knowledge Gaps
- **116 isolated node(s):** `glass-box`, `Native Windows Streamlit Application`, `HistGradientBoosting Baseline`, `Persisted Run Artifacts`, `California Housing Dataset` (+111 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 217 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Persist and display phase (partially visible)` and `Persist predictor`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `load_sample()` connect `Dataset Catalog` to `EDA and Visualization`, `Training Pages and Uploads`, `Classifier Smoke Checks`, `Dataset Smoke Tests`, `GPU Fine Tuning Smoke`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `run_mitra()` connect `MITRA Model Runner` to `GPU Fine Tuning Smoke`, `Training Pages and Uploads`, `Classifier Smoke Checks`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `current_data()` connect `Training Pages and Uploads` to `Streamlit Pages`, `EDA and Visualization`, `Dataset Catalog`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `glass-box`, `Native Windows Streamlit Application`, `HistGradientBoosting Baseline` to the rest of the system?**
  _116 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Project Documentation` be split into smaller, more focused modules?**
  _Cohesion score 0.06423034330011074 - nodes in this community are weakly interconnected._
- **Should `EDA and Visualization` be split into smaller, more focused modules?**
  _Cohesion score 0.09672830725462304 - nodes in this community are weakly interconnected._
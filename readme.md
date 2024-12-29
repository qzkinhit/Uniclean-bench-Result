# UnicleanResult: A Benchmark Repository for Data Cleaning Performance

## Overview
**UnicleanResult** is a repository dedicated to showcasing the performance of Uniclean, a state-of-the-art data cleaning system. While we cannot open source the system code due to commercial confidentiality, this repository provides comprehensive performance metrics and detailed cleaning results, establishing a benchmark for data cleaning. This enables researchers to evaluate and compare the performance of their own cleaning systems on various real-world datasets. The benchmark allows for direct comparisons between Uniclean’s performance, other baseline cleaning systems, and new approaches, offering a standardized framework for assessing data cleaning effectiveness across diverse datasets.

The repository includes:
- **Real-world native datasets** used by Uniclean for testing.
- **Cleaned datasets** that have been processed by Uniclean.
- **Cleaning logs** generated during the Uniclean cleaning process.
- **Baseline performance logs** for comparison with Uniclean’s results.
- An **evaluation script** (`evaluateResult.py`) that calculates various performance metrics, providing an objective assessment of the cleaning effectiveness.

## Dataset Information

The following table summarizes the datasets used in this repository, including their error types and dimensions:

| Dataset  | Error Type     | Shape        | Link                                                                             |
|----------|----------------|--------------|----------------------------------------------------------------------------------|
| Hospital | T, VAD         | 1,000 × 20   | [datasets/original_datasets/1_hospital](datasets_and_rules/original_datasets/1_hospital) |
| Flights  | MV, FI, VAD    | 2,376 × 7    | [datasets/original_datasets/2_flights](datasets_and_rules/original_datasets/2_flights)   |
| Beers    | MV, FI, VAD    | 2,410 × 111  | [datasets/original_datasets/3_beers](datasets_and_rules/original_datasets/3_beers)       |
| Rayyan   | MV, T, FI, VAD | 1,000 × 11   | [datasets/original_datasets/4_rayyan](datasets_and_rules/original_datasets/4_rayyan)     |
| Tax      | T, FI, VAD     | 200,000 × 15 | [datasets/original_datasets/5_tax](datasets_and_rules/original_datasets/5_tax)           |
| Soccer   | T, VAD         | 200,000 × 15 | [datasets/original_datasets/6_soccer](datasets_and_rules/original_datasets/6_soccer)     |

**Error Type Abbreviations:**
- **T**: Typographical errors
- **MV**: Missing values
- **FI**: Format inconsistencies
- **VAD**: Violated attribute dependencies


## Running Uniclean’s Cleaning Performance Test

To evaluate Uniclean’s cleaning performance, run the `run.sh` script. This script automates the cleaning process across all datasets and saves performance logs in the `Uniclean_logs/` directory.

### Usage
```bash
# Give execution permissions
chmod +x run.sh

# Run the script
./run.sh
```

The `run.sh` script iterates over each dataset in the `datasets/original_datasets/` directory, processes it with Uniclean, and logs the results. Each dataset has its specific configuration, including `mse_attributes` (attributes for Mean Squared Error calculation) and `elapsed_time` parameters. The results of each dataset’s cleaning process are saved in the corresponding subdirectory within `Uniclean_logs/`.

## Repository Structure
- `datasets_and_rules/`:real word datasets、inject error datasets and their cleaning rules:
  - `artificial_error_datasets/`:Contains datasets with artificially injected errors in eight different proportions (ranging from 0.25% to 2%) for controlled experiments and benchmarking. This folder also includes the *BART script* used for injecting these errors into the datasets.
  - `original_datasets/`: Contains real-world datasets in their native (uncleaned) form.
- `Uniclean_cleaned_data/`: Datasets that have been cleaned by Uniclean.
  - `artificial_error_cleaned_data/`:Uniclean-cleaned versions of the artificially injected error datasets.
  - `original_error_cleaned_data/`:Uniclean-cleaned  versions of the real-world datasets containing native errors.
- `Uniclean_cleaner_workflow_logs/`: Logs generated during the Uniclean cleaning process and Cleaner attributes dependencies for each dataset.
  - `artificial_error_cleaner_workflow_logs/`: Step-by-step workflow logs for datasets that had artificial errors (in different proportions).
  - `original_error_cleaner_workflow_logs/`:Step-by-step workflow logs for real-world datasets with native errors.
- `Uniclean_results/`: Contains the final outputs and performance metrics from Uniclean’s data cleaning for each dataset.
  - `artificial_error_results/`:Final outputs and metrics (e.g., accuracy, F1 score) from Uniclean’s cleaning for datasets that had artificially injected errors in different proportions.
  - `original_error_results/`:Final outputs and metrics from Uniclean’s cleaning for real-world datasets containing native errors.
- `baseline_cleaning_systems_logs/`: Logs documenting the performance of baseline systems on the same datasets, enabling a direct comparison with Uniclean’s results.
  - `artificial_error_datasets/`:Stores log files showing how baseline systems perform on datasets with artificial errors.
    - **File Naming Format**: `[dataset_name]_[cleaning_system_name]_nwcpk_[error_proportion].log`
    - Example: `1_hospitals_raha_baran_nwcpk_1.log`
  - `original_datasets/`:Stores log files showing how baseline systems perform on real-world datasets with native errors.
    - **File Naming Format**: `[dataset_name]_ori_[cleaning_system_name]_[the actual size of the dataset (if it is not in its original size)].log`
    - Example: `1_hospital_ori_baran.log`
- `baseline_cleaning_systems_results/`: Final results and performance metrics of baseline systems on the same datasets.
  - `artificial_error_datasets/`:Contains overall performance metrics (e.g., accuracy, recall, F1 score) of baseline systems on artificially injected error datasets.
    - **Folder Naming Format**: `[dataset_name]_nwcpk_[error_proportion]`
    - Example: `1_hospitals_nwcpk_1`
  - `original_datasets/`:Contains overall performance metrics of baseline systems on real-world datasets with native errors.
    - **Folder Naming Format**: `[dataset_name]_[the actual size of the dataset (if it is not in its original size)]_ori`
    - Example: `1_hospital_ori`
- `baseline_cleaned_data/`:Datasets that have been cleaned by baseline systems.
  - `artificial_error_datasets/`:Baseline-cleaned versions of artificially injected error datasets.
    - **File Naming Format**: `[dataset_name]_[error_proportion]_cleaned_by_[cleaning_system_name].csv`
    - Example: `1_hospitals_1_cleaned_by_baran.csv`
  - `original_datasets/`:Baseline-cleaned versions of real-world datasets with native errors.
    - **File Naming Format**: `[dataset_name][the actual size of the dataset (if it is not in its original size)]_cleaned_by_[cleaning_system_name].csv`
    - Example: `1_hospital_cleaned_by_baran.csv`
- `evaluate_result.py`: A script that computes performance metrics for data cleaning, such as accuracy, recall, F1 score, and error reduction rate, allowing comprehensive evaluation of data cleaning effectiveness.
- `get_holoclean_table.py` A script that transforms datasets into the Holoclean-compatible input CSV format. It transposes data and ensures compliance with Holoclean's required schema for further data cleaning tasks.
- `get_error_num.py` A script that compares dirty data with clean data to compute the number of erroneous cells and entries. It provides a detailed analysis of the extent of errors, facilitating error quantification and benchmarking.
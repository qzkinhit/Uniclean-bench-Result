# UnicleanResult: A Benchmark Repository for Data Cleaning Performance

## Overview
**UnicleanResult** is a repository dedicated to showcasing the performance of Uniclean, a state-of-the-art data cleaning system. In addition to presenting Uniclean's results, this repository establishes a benchmark for data cleaning, enabling researchers to evaluate and compare the performance of their own cleaning systems on various real-world datasets. This benchmark allows for direct comparisons between Uniclean’s performance, other baseline cleaning systems, and new approaches, providing a standardized framework for assessing data cleaning effectiveness across diverse datasets.

The repository includes:
- **Real-world native datasets** used by Uniclean for testing.
- **Cleaned datasets** that have been processed by Uniclean.
- **Cleaning logs** generated during the Uniclean cleaning process.
- **Baseline performance logs** for comparison with Uniclean’s results.
- An **evaluation script** (`evaluateResult.py`) that calculates various performance metrics, providing an objective assessment of the cleaning effectiveness.

## Repository Structure

- `RealWorldDataSet/`: Contains real-world datasets in their native (uncleaned) form.
- `Uniclean_cleaned_data/`: Datasets that have been cleaned by Uniclean.
- `Uniclean_logs/`: Logs generated during the Uniclean cleaning process, detailing operations and results for each dataset.
- `baseline_logs/`: Logs documenting the performance of baseline systems on the same datasets, enabling a direct comparison with Uniclean’s results.
- `evaluateResult.py`: A script that computes performance metrics for data cleaning, such as accuracy, recall, F1 score, and error reduction rate, allowing comprehensive evaluation of data cleaning effectiveness.

## Dataset Information

The following table summarizes the datasets used in this repository, including their error types and dimensions:

| Dataset | Error Type | Shape | Link |
|---------|------------|-------|------|
| Hospital | T, VAD | 1,000 × 20 | [RealWorldDataSet/1_hospital](./RealWorldDataSet/1_hospital) |
| Flights  | MV, FI, VAD | 2,376 × 7 | [RealWorldDataSet/2_flights](./RealWorldDataSet/2_flights) |
| Beers    | MV, FI, VAD | 2,410 × 111 | [RealWorldDataSet/3_beers](./RealWorldDataSet/3_beers) |
| Rayyan   | MV, T, FI, VAD | 1,000 × 11 | [RealWorldDataSet/4_rayyan](./RealWorldDataSet/4_rayyan) |
| Tax      | T, FI, VAD | 200,000 × 15 | [RealWorldDataSet/5_tax](./RealWorldDataSet/5_tax50k) |

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

The `run.sh` script iterates over each dataset in the `RealWorldDataSet/` directory, processes it with Uniclean, and logs the results. Each dataset has its specific configuration, including `mse_attributes` (attributes for Mean Squared Error calculation) and `elapsed_time` parameters. The results of each dataset’s cleaning process are saved in the corresponding subdirectory within `Uniclean_logs/`.
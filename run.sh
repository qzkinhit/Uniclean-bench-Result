#!/bin/bash

# chmod +x run.sh
# ./run.sh

# Define datasets, their specific mse_attributes, and pre-calculated elapsed times
declare -A mse_attributes
declare -A elapsed_times

mse_attributes=(
    ["1_hospital"]="Score"
    ["2_flights"]=""
    ["3_beers"]="abv ibu"
    ["4_rayyan"]=""
    ["5_tax"]="rate"
)

elapsed_times=(
    ["1_hospital"]="108.115330875"
    ["2_flights"]="84.591757375"
    ["3_beers"]="31.24922154099"
    ["4_rayyan"]="52.377824583"
    ["5_tax"]="12589.6389"
)

# Define base paths
dirty_base_path="./RealWorldDataSet"
clean_base_path="./RealWorldDataSet"
cleaned_base_path="./Uniclean_cleaned_data"
output_base_path="./Uniclean_logs"

# Define datasets to process
datasets=("1_hospital" "2_flights" "3_beers" "4_rayyan" "5_tax")

# Loop through each dataset and execute the main Python script
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    # Define paths for the current dataset
    dirty_path="${dirty_base_path}/${dataset}/dirty_index.csv"
    clean_path="${clean_base_path}/${dataset}/clean_index.csv"
    cleaned_path="${cleaned_base_path}/${dataset}_cleaned_by_uniclean.csv"
    output_path="${output_base_path}/${dataset}"

    # Define task-specific settings
    task_name="${dataset}"

    # Get mse_attributes and elapsed time for the current dataset
    mse_attr="${mse_attributes[$dataset]}"
    elapsed_time="${elapsed_times[$dataset]}"

    # Construct command with mse_attributes and elapsed_time if they exist
    if [ -n "$mse_attr" ]; then
        python3 main.py \
            --dirty_path "$dirty_path" \
            --clean_path "$clean_path" \
            --cleaned_path "$cleaned_path" \
            --output_path "$output_path" \
            --task_name "$task_name" \
            --index_attribute "index" \
            --mse_attributes $mse_attr \
            --elapsed_time "$elapsed_time" > "${output_path}/output.log"
    else
        python3 main.py \
            --dirty_path "$dirty_path" \
            --clean_path "$clean_path" \
            --cleaned_path "$cleaned_path" \
            --output_path "$output_path" \
            --task_name "$task_name" \
            --index_attribute "index" \
            --elapsed_time "$elapsed_time" > "${output_path}/output.log"
    fi

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Processing failed for dataset: $dataset"
        exit 1
    fi

    echo "Completed processing for dataset: $dataset"
    echo "--------------------------------------------------"
done

echo "All datasets processed."
#!/bin/bash

# Define the directory where the engine files are located
batch="$1"
engine_directory="./Bragg_Onnx_MOBJ_engines_int8_b${batch}"

# Create an output CSV file
output_file="${engine_directory}_latency_results.csv"
echo "Engine File,Mean Latency (ms),Max Latency (ms),Min Latency (ms),90th Percentile Latency (ms),95th Percentile Latency (ms),99th Percentile Latency (ms)" > "$output_file"

# Directory to save command outputs
output_directory="${engine_directory}_command_outputs"
mkdir -p "$output_directory"

# Loop through each engine file in the directory
for engine_file in "${engine_directory}"/*.engine; do
    # Extract the file name without extension
    filename=$(basename -- "$engine_file")
    filename_no_ext="${filename%.*}"

    # Generate the command using the engine file
    command="trtexec --useCudaGraph --useSpinWait --loadEngine=${engine_file}"

    # Run the command and save output to log file
    log_file="${output_directory}/${filename_no_ext}_output.log"
    $command &> "$log_file"

    # Extract desired output from the log file
    output=$(grep -m 1 'Latency:.*percentile(99%)' "$log_file")

    # Extract individual latency values
    mean_latency=$(echo "$output" | grep -o 'mean = [0-9.]*' | cut -d' ' -f3)
    max_latency=$(echo "$output" | grep -o 'max = [0-9.]*' | cut -d' ' -f3)
    min_latency=$(echo "$output" | grep -o 'min = [0-9.]*' | cut -d' ' -f3)
    percentile_90_latency=$(echo "$output" | grep -o 'percentile(90%) = [0-9.]*' | cut -d' ' -f3)
    percentile_95_latency=$(echo "$output" | grep -o 'percentile(95%) = [0-9.]*' | cut -d' ' -f3)
    percentile_99_latency=$(echo "$output" | grep -o 'percentile(99%) = [0-9.]*' | cut -d' ' -f3)

    # Write results to CSV
    echo "${filename_no_ext},${mean_latency},${max_latency},${min_latency},${percentile_90_latency},${percentile_95_latency},${percentile_99_latency}" >> "$output_file"
done


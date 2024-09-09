#!/bin/bash
# Ramyad's Comment: Mixing create and run engine to save time. trt already does an execution in the end of the enigne creations

# Define the directories where the ONNX files are located and where the engines will be saved
onnx_directory="./Ptycho_Onnx_SOBJ"
batch="$1"
engine_directory="Ptycho_Onnx_SOBJ_engines_int8_b${batch}"

# Create the engine directory if it doesn't exist
mkdir -p "$engine_directory"

# Create an output CSV file
output_file="${engine_directory}_latency_results.csv"
echo "Engine File,Mean Latency (ms),Max Latency (ms),Min Latency (ms),90th Percentile Latency (ms),95th Percentile Latency (ms),99th Percentile Latency (ms)" > "$output_file"

# Directory to save command outputs
output_directory="${engine_directory}_command_outputs"
mkdir -p "$output_directory"


# Loop through each file in the ONNX directory
for onnx_file in "${onnx_directory}"/*.onnx; do
    # Extract the file name without extension
    filename=$(basename -- "$onnx_file")
    filename_no_ext="${filename%.*}"
    
    # Generate the path for the engine file
    engine_path="${engine_directory}/${filename_no_ext}-int8.engine"
    
    # Generate the command using the file name and engine path
    command="trtexec --onnx=${onnx_file} --saveEngine=${engine_path} --int8 --optShapes=input:${batch}x1x64x64"
    
    # Run the command and save output to log file
    echo "Running command: $command"
    log_file="${output_directory}/${filename_no_ext}_output.log"
    $command | tee "$log_file"

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

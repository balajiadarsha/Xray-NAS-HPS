#!/bin/bash

# Define the directories where the ONNX files are located and where the engines will be saved

onnx_directory="./Bragg_corner_models"
batch="$1"
engine_directory="Bragg_Onnx_corner_engines_int8_b${batch}"

# Create the engine directory if it doesn't exist
mkdir -p "$engine_directory"

# Loop through each file in the ONNX directory
for onnx_file in "${onnx_directory}"/*.onnx; do
    # Extract the file name without extension
    filename=$(basename -- "$onnx_file")
    filename_no_ext="${filename%.*}"
    
    # Generate the path for the engine file
    engine_path="${engine_directory}/${filename_no_ext}-int8.engine"
    
    # Generate the command using the file name and engine path
    command="trtexec --onnx=${onnx_file} --optShapes="/Flatten_output_0":${batch}x121 --int8 --saveEngine=${engine_path}"
    
    # Run the command
    echo "Running command: $command"
    eval $command
done

onnx_directory="/mnt/nvme/anl-profiling-data/PtychoNN_corner_models"
batch="$1"
engine_directory="Ptycho_Onnx_corner_engines_int8_b${batch}"

# Create the engine directory if it doesn't exist
mkdir -p "$engine_directory"

# Loop through each file in the ONNX directory
for onnx_file in "${onnx_directory}"/*.onnx; do
    # Extract the file name without extension
    filename=$(basename -- "$onnx_file")
    filename_no_ext="${filename%.*}"
    
    # Generate the path for the engine file
    engine_path="${engine_directory}/${filename_no_ext}-int8.engine"
    
    # Generate the command using the file name and engine path
    command="trtexec --onnx=${onnx_file} --saveEngine=${engine_path} --int8 --optShapes=input:${batch}x1x64x64"
    
    # Run the command
    echo "Running command: $command"
    eval $command
done

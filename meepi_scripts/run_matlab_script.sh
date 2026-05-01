#!/bin/bash
# Unload and load correct MATLAB versions
#module unload matlab/2022a
module load matlab/2023a
module load spm/12

# Check if two arguments are provided (subjectID and sessionID)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <subjectID> <sessionID>"
    exit 1
fi

subjectID=$1
sessionID=$2

# Define the directories for input and output
input_folder="/cbica/projects/pafin/dset/sub-${subjectID}/ses-${sessionID}/anat"
baseoutput_folder="/cbica/projects/pafin/projects/qsm-validation/derivatives/meepi/sub-${subjectID}/ses-${sessionID}"

# Create the output directory if it doesn't exist
mkdir -p "${baseoutput_folder}"
output_folder="${baseoutput_folder}"
# Specify the directory where the MATLAB function is located
MATLAB_SCRIPT_DIR="/cbica/projects/pafin/projects/qsm-validation/software/sepia"
# Run the MATLAB script
matlab -nodisplay -nosplash -r "addpath(genpath('$MATLAB_SCRIPT_DIR')); try; run_chisep_script('$input_folder', '$output_folder'); catch e; disp(e.message); end; exit;"

# Check if MATLAB succeeded and handle errors
if [ $? -ne 0 ]; then
    echo "Error processing subject $subjectID session $sessionID"
else
    echo "Successfully processed subject $subjectID session $sessionID"
fi

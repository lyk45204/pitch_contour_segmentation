# Pitch Contour Segmentation using HMM

## Overview
This MATLAB project focuses on the automatic segmentation of pitch contours into three primary types of regions: transitory, steady, and vibrato (modulation), utilizing Hidden Markov Models (HMMs) alongside Frequency Domain Models (FDM) for enhanced analysis. The project segments pitch contours by analyzing quantum-level attributes, characterized by local extremities within the pitch contour, to understand the nuances of musical performance.

## Features
- **Quantum-Level Analysis**: Detailed examination of pitch contours through the lens of quantums, with each quantum defined by its duration and extent.
- **Advanced Modeling Techniques**: Utilizes both Hidden Markov Models (HMM) and Frequency Domain Models (FDM) for comprehensive pitch contour segmentation.
- **Observation Likelihood Distribution**: Employs Kernel Density Estimation (KDE) for smoothing, with optimization through K-fold cross-validation to refine the analysis.
- **Comprehensive Evaluation Package**: Includes tools for model performance evaluation and pitch state validation.

## Dependencies
- MATLAB (Version 2018b or later recommended)
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pitch_contour_segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pitch_contour_segmentation
   ```

## Usage
1. Load your pitch contour data into MATLAB.
2. Run the main segmentation script (replace `[script_name]` with the actual script name):
   ```matlab
   [script_name]
   ```

## Data Format
Describe the expected data format for the pitch contour input.

## Training Method
The model is trained using a supervised approach. Training data should be prepared as follows:
- get the pitch track of the audio
- annotate the pitch contour patterns regions.

## Contributing
Contributions to this project are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b [branch_name]`.
3. Make your changes and commit them: `git commit -m '[commit_message]'`.
4. Push to the original branch: `git push origin [project_name]/[location]`.
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## License


## Contact
- Yukun Li
- yukun.li@qmul.ac.uk



```


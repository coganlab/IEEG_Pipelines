# IEEG_Pipelines

A repo of current preprocessing pipelines for the [Cogan Lab](https://www.coganlab.org/)

[![Brain](./docs/images/brain_rot.gif)](https://www.coganlab.org/)

## Documentation

[![Documentation Status](https://readthedocs.org/projects/ieeg-pipelines/badge/?version=latest)](https://ieeg-pipelines.readthedocs.io/en/latest/?badge=latest)

[Lab Wiki](https://coganlab.pages.oit.duke.edu/wiki//)

## Pipeline Functionality

[![Python (3.10) on Windows/Linux](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/Python-CI.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/Python-CI.yml)

[![MATLAB latest](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml)

[![codecov](https://codecov.io/gh/coganlab/IEEG_Pipelines/branch/main/graph/badge.svg?token=X4IAFGOBGN)](https://codecov.io/gh/coganlab/IEEG_Pipelines)

## Installation

### MATLAB

1. Install MATLAB
2. Clone this repository into your userpath (`Documents/MATLAB` by default)
3. Run commands:

    ```MATLAB
    path = fullfile(userpath, 'IEEG_Pipelines', 'MATLAB');
    addpath(genpath(path));
    ```

### Python

Version 3.10 supported

#### Conda

1. Install Anaconda
2. Create an anaconda environment with python and pip packages installed
    
     ```bash
     conda create -n <YOUR_NAME> python<3.13 pip
     ```
3. Activate the environment

    ```bash
    conda activate <YOUR_NAME>
    ```
   
4. Run

    ```bash
    pip install ieeg
    ```

#### [Pip](https://pypi.org/project/ieeg/)

1. Install Python
2. Run:

    ```bash
    python -m venv <PATH TO VENV>/<YOUR_NAME>
    source activate <PATH TO VENV>/<YOUR_NAME>
    python -m pip install ieeg
    ```
   
## Usage

### MATLAB (INCOMPLETE)

1. Load `.dat` file using [convert_OpenE_rec2mat.m](MATLAB/ieeg%20file%20reading/convert_OpenE_rec2mat.m)
2. Create the ieeg data structure from the [ieegStructClass.m](MATLAB/ieegClassDefinition/ieegStructClass.m)
3. `TBD`

### Python ([INCOMPLETE](https://github.com/orgs/coganlab/projects/7))

1. Load BIDS files from BIDS directory using [`pybids`](https://bids-standard.github.io/pybids/)
    
    ```python
    from bids import BIDSLayout
    import ieeg
    layout = BIDSLayout(<BIDS_root>)
    data = ieeg.io.raw_from_layout(layout)
    ```
2. [Perform line noise filtering](https://ieeg-pipelines.readthedocs.io/en/latest/auto_examples/plot_clean.html)

3. [Check Spectrograms](https://ieeg-pipelines.readthedocs.io/en/latest/auto_examples/plot_spectrograms.html)

4. [Plot the high gamma responses](https://ieeg-pipelines.readthedocs.io/en/latest/auto_examples/plot_HG.html)

5. [Run the cluster correction and permutation test](https://ieeg-pipelines.readthedocs.io/en/latest/auto_examples/plot_stats.html)

# IEEG_Pipelines

A repo of current preprocessing pipelines for the [Cogan Lab](https://www.coganlab.org/)

## Documentation

[![Documentation Status](https://readthedocs.org/projects/ieeg-pipelines/badge/?version=latest)](https://ieeg-pipelines.readthedocs.io/en/latest/?badge=latest)

[Lab Wiki](https://coganlab.pages.oit.duke.edu/wiki//)

## Pipeline Functionality

[![Python (3.10) on Windows/Linux](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/Conda-CI.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/Conda-CI.yml)

[![MATLAB latest](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml)

[![codecov](https://codecov.io/gh/coganlab/IEEG_Pipelines/branch/main/graph/badge.svg?token=X4IAFGOBGN)](https://codecov.io/gh/coganlab/IEEG_Pipelines)

## Installation

### MATLAB

1. Install MATLAB
2. Clone this repository into your userpath (`Documents/MATLAB` by default)
3. Run commands:

    ```(MATLAB)
    path = fullfile(userpath, 'IEEG_Pipelines', 'MATLAB');
    addpath(genpath(path));
    ```

### Python

Version 3.10 supported

#### Conda

1. Install Anaconda
2. Clone this repository
3. Open a terminal and `cd` into this repo's `Python` directory
4. Run this command:

    ```(bash)
    conda env create -f envs/environment.yml
    ```

5. When it is finished installing run `conda activate preprocess` to activate the environment

#### Pip

1. Install Python
2. Clone this repository
3. Open a terminal and `cd` into this repo's `Python` directory
4. Run:

    ```(bash)
    python -m venv <PATH TO VENV>/preprocess
    python -m pip install -r envs/requirements.txt -e <PATH TO VENV>/preprocess
    ```

5. When it is finished installing run `source activate <PATH TO VENV>/preprocess` to activate the environment

## Usage

### MATLAB ([INCOMPLETE](https://github.com/coganlab/IEEG_Pipelines/issues/21#issue-1229145282))

1. Load `.dat` file using [convert_OpenE_rec2mat.m](MATLAB/ieeg%20file%20reading/convert_OpenE_rec2mat.m)
2. Create the ieeg data structure from the [ieegStructClass.m](MATLAB/ieegClassDefinition/ieegStructClass.m)
3. `TBD`

### Python ([INCOMPLETE](https://github.com/coganlab/IEEG_Pipelines/issues/22#issue-1229152846))

1. Load BIDS files from BIDS directory using `pybids` and [preProcess.py](Python/PreProcess/preProcess.py)
    
    ```(python)
    from bids import BIDSLayout
    import preProcess as pre
    layout = BIDSLayout(BIDS_root)
    data = pre.raw_from_layout(layout)
    ```
2. Perform line noise filtering

    ```(python)
    filt = pre.line_filter(data)
    ```
3. `TBD`

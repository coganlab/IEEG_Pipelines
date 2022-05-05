# IEEG_Pipelines
A repo of current preprocessing pipelines for the [Cogan Lab](https://www.coganlab.org/)

## Pipeline Functionality

[![Python (3.8, 3.9) on Windows/Linux](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/python-package.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/python-package.yml)

[![MATLAB latest](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml/badge.svg)](https://github.com/coganlab/IEEG_Pipelines/actions/workflows/MATLAB-CI.yml)

## Installation

### MATLAB
1. Install MATLAB
2. Clone this repository into your userpath (`Documents/MATLAB` by default)
3. Run commands:
```
path = fullfile(userpath, 'IEEG_Pipelines', 'MATLAB');
addpath(genpath(path));
```

### Python
1. Install Anaconda
2. Clone this repository
3. open a terminal and `cd` into this repo's `Python` directory
4. Run this command:
```
conda env create -f envs/environment.yml
```
5. When it is finished installing run `conda activate preprocessing` to activate the environment
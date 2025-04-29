# This file initializes the _fast module and makes the necessary functions available

# Import functions from submodules
from ieeg.calc._fast.ufuncs import mean_diff, t_test
from ieeg.calc._fast.mixup import mixupnd, normnd
from ieeg.calc._fast.permgt import permgtnd

# Export these functions
__all__ = ['mean_diff', 't_test', 'mixupnd', 'normnd', 'permgtnd']
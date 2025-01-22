import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the path to the submodule directory
compat_path = os.path.join(current_dir, 'array-api-compat')
sys.path.insert(0, compat_path)
extras_path = os.path.join(current_dir, 'array-api-extra', 'src')
sys.path.insert(0, extras_path)


# Now you can import the submodule
import array_api_compat
import array_api_extra

__all__ = ['array_api_compat', 'array_api_extra']

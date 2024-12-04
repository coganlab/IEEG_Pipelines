# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme, sphinx_gallery
from docs._resetter import Resetter
sys.path.insert(0, os.path.abspath('..'))

import warnings
import mne
import os
from mne.viz import Brain # noqa, needed for mne.viz._brain._BrainScraper



# -- Project information -----------------------------------------------------

project = 'IEEG_Pipelines'
copyright = '2024, Aaron Earle-Richardson'
author = 'Aaron Earle-Richardson'

# The full version, including alpha/beta/rc tags
release = '0.6.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['myst_parser',
              'sphinx_gallery.gen_gallery',
              'sphinxcontrib.matlab',
              'sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'sphinx.ext.linkcode',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx_copybutton']

# -- TOC configuration -------------------------------------------------------

toc_object_entries = False

# -- Sphinx-Copybutton configuration -----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for plot_directive ----------------------------------------------

# Adapted from SciPy
plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False
font_size = 13 * 72 / 96.0  # 13 px
plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (6, 5),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# -- Sphinx-gallery configuration --------------------------------------------

scrapers = (
    "matplotlib",
    mne.gui._GUIScraper(),
    mne.viz._brain._BrainScraper(),
    "pyvista",
    mne.report._ReportScraper(),
    mne.viz._scraper._MNEQtBrowserScraper(),
)

os.environ["_MNE_BUILDING_DOC"] = "true"
os.environ["MNE_3D_OPTION_ANTIALIAS"] = "false"
mne.viz.set_3d_backend("pyvistaqt")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyvista
    if os.environ.get("READTHEDOCS") == "True":
        pyvista.start_xvfb()
        pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'ieeg': None,
    },
    "recommender": {"enable": True,
                    "n_examples": 3},
    # Add ability to link to mini-gallery under each function/class
    # directory where function/class granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and ieeg in a tuple of strings.
    'expected_failing_examples': ['../examples/plot_data.py'],
    "plot_gallery": "True",
    'doc_module': ('sphinx_gallery', 'ieeg'),
    "reset_modules": ("matplotlib", Resetter()),  # called w/each script
    "reset_modules_order": "both",
    "image_scrapers": scrapers,
    "show_memory": not sys.platform.startswith(("win", "darwin")),
    "thumbnail_size": (160, 112),
}

default_role = 'py:obj'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autosummary_generate = True
notebook_images = True
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

autodoc_typehints = 'both'
add_module_names = False

matlab_src_dir = os.path.abspath('../MATLAB')
matlab_auto_link = True

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'mne': ('https://mne.tools/stable/', None),
                       'mne_bids': ('https://mne.tools/mne-bids/stable/',
                                    None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference/',
                                 None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'bids': ('https://bids-standard.github.io/pybids/',
                                None),
                       'joblib': ('https://joblib.readthedocs.io/en/latest/',
                                  None),
                       "sklearn": ("http://scikit-learn.org/dev", None),
                       "numba": ("https://numba.readthedocs.io/en/latest", None),
                       "nibabel": ("https://nipy.org/nibabel", None),
                       "mne-gui-addons": ("https://mne.tools/mne-gui-addons", None),
                       "pyvista": ("https://docs.pyvista.org", None),
                       "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
                       "dipy": ("https://docs.dipy.org/stable", None),
                       "pyqtgraph": ("https://pyqtgraph.readthedocs.io/en/latest/", None),
                       }


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/coganlab/IEEG_Pipelines/%s.py" % filename


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['tests', 'setup']


# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
modindex_common_prefix = ["ieeg"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "logo_only": False,
}
html_logo = "./images/brain_logo_blue.png"
html_favicon = "./images/favicon.ico"

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/
    # {{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    "github_user": "coganlab",
    "github_repo": "IEEG_Pipelines",
    "github_version": "main/",
    "conf_py_path": "docs/",
}

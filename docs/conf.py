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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'IEEG_Pipelines'
copyright = '2023, Aaron Earle-Richardson'
author = 'Aaron Earle-Richardson'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['myst_nb',
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
              'sphinx.ext.ifconfig',
              'sphinx.ext.githubpages',
              'sphinx.ext.todo']
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'ieeg': None,
    },
    # Add ability to link to mini-gallery under each function/class
    # directory where function/class granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and ieeg in a tuple of strings.
    'doc_module': ('sphinx_gallery', 'ieeg'),
    'filename_pattern': '/plot_'
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

matlab_src_dir = os.path.abspath('../MATLAB')
matlab_auto_link = True


# nb_execution_mode = 'off'

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
                       }


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/coganlab/IEEG_Pipelines/%s.py" % filename


def setup(app):
    # to hide/show the prompt in code examples:
    app.add_js_file("js/copybutton.js")


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['tests']


# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
modindex_common_prefix = ["ieeg."]

pygments_style = "sphinx"
smartquotes = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    # 'includehidden': False,
    "collapse_navigation": False,
    "navigation_depth": 3,
    "logo_only": False,
}
html_logo = "./images/brain_logo.png"
html_favicon = "./images/favicon.ico"

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/
    # {{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    "github_user": "coganlab",
    "github_repo": "IEEG_Pipelines",
    "github_version": "main/docs/",
}

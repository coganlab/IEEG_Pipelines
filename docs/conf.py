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


import time
import warnings
import gc
import mne
import os
import matplotlib
from mne.utils import (
    _assert_no_instances, # noqa, analysis:ignore
    sizeof_fmt,
)
from mne.viz import Brain # noqa, needed for mne.viz._brain._BrainScraper
import matplotlib


class Resetter(object):
    """Simple class to make the str(obj) static for Sphinx build env hash."""

    def __init__(self):
        self.t0 = time.time()

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, gallery_conf, fname, when):
        import matplotlib.pyplot as plt

        try:
            from pyvista import Plotter  # noqa
        except ImportError:
            Plotter = None  # noqa
        try:
            from pyvistaqt import BackgroundPlotter  # noqa
        except ImportError:
            BackgroundPlotter = None  # noqa
        try:
            from vtkmodules.vtkCommonDataModel import vtkPolyData  # noqa
        except ImportError:
            vtkPolyData = None  # noqa
        try:
            from mne_qt_browser._pg_figure import MNEQtBrowser
        except ImportError:
            MNEQtBrowser = None
        from mne.viz.backends.renderer import backend

        _Renderer = backend._Renderer if backend is not None else None
        # in case users have interactive mode turned on in matplotlibrc,
        # turn it off here (otherwise the build can be very slow)
        plt.ioff()
        plt.rcParams["animation.embed_limit"] = 40.0
        plt.rcParams["figure.raise_window"] = False
        # https://github.com/sphinx-gallery/sphinx-gallery/pull/1243#issue-2043332860
        plt.rcParams["animation.html"] = "html5"
        # neo holds on to an exception, which in turn holds a stack frame,
        # which will keep alive the global vars during SG execution
        try:
            import neo

            neo.io.stimfitio.STFIO_ERR = None
        except Exception:
            pass
        gc.collect()

        # Agg does not call close_event so let's clean up on our own :(
        # https://github.com/matplotlib/matplotlib/issues/18609
        mne.viz.ui_events._cleanup_agg()
        assert len(mne.viz.ui_events._event_channels) == 0, list(
            mne.viz.ui_events._event_channels
        )

        when = f"mne/conf.py:Resetter.__call__:{when}:{fname}"
        # Support stuff like
        # MNE_SKIP_INSTANCE_ASSERTIONS="Brain,Plotter,BackgroundPlotter,vtkPolyData,_Renderer" make html-memory  # noqa: E501
        # to just test MNEQtBrowser
        skips = os.getenv("MNE_SKIP_INSTANCE_ASSERTIONS", "").lower()
        prefix = ""
        if skips not in ("true", "1", "all"):
            prefix = "Clean "
            skips = skips.split(",")
            if "brain" not in skips:
                _assert_no_instances(mne.viz.Brain, when)  # calls gc.collect()
            if Plotter is not None and "plotter" not in skips:
                _assert_no_instances(Plotter, when)
            if BackgroundPlotter is not None and "backgroundplotter" not in skips:
                _assert_no_instances(BackgroundPlotter, when)
            if vtkPolyData is not None and "vtkpolydata" not in skips:
                _assert_no_instances(vtkPolyData, when)
            if "_renderer" not in skips:
                _assert_no_instances(_Renderer, when)
            if MNEQtBrowser is not None and "mneqtbrowser" not in skips:
                # Ensure any manual fig.close() events get properly handled
                from mne_qt_browser._pg_figure import QApplication

                inst = QApplication.instance()
                if inst is not None:
                    for _ in range(2):
                        inst.processEvents()
                _assert_no_instances(MNEQtBrowser, when)
        # This will overwrite some Sphinx printing but it's useful
        # for memory timestamps
        if os.getenv("SG_STAMP_STARTS", "").lower() == "true":
            import psutil

            process = psutil.Process(os.getpid())
            mem = sizeof_fmt(process.memory_info().rss)
            print(f"{prefix}{time.time() - self.t0:6.1f} s : {mem}".ljust(22))


# -- Project information -----------------------------------------------------

project = 'IEEG_Pipelines'
copyright = '2024, Aaron Earle-Richardson'
author = 'Aaron Earle-Richardson'

# The full version, including alpha/beta/rc tags
release = '0.1'


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
    if os.getenv("PYVISTA_OFF_SCREEN", "false").lower() == "true":
        pyvista.start_xvfb()
        pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True

sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    'reference_url': {
        'ieeg': None,
    },
    "recommender": {"enable": True},
    # Add ability to link to mini-gallery under each function/class
    # directory where function/class granular galleries are stored
    'backreferences_dir': 'gen_modules/backreferences',
    # Modules for which function/class level galleries are created. In
    # this case sphinx_gallery and ieeg in a tuple of strings.
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
exclude_patterns = ['tests']


# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
modindex_common_prefix = ["ieeg"]

pygments_style = "sphinx"
# smartquotes = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    # 'includehidden': False,
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

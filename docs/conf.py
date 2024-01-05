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
import numpy as np
import os
from mne.utils import (
    _assert_no_instances, # noqa, analysis:ignore
    sizeof_fmt,
)

_np_print_defaults = np.get_printoptions()


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
        reset_warnings(gallery_conf, fname)
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


# -- Warnings management -----------------------------------------------------


def reset_warnings(gallery_conf, fname):
    """Ensure we are future compatible and ignore silly warnings."""
    # In principle, our examples should produce no warnings.
    # Here we cause warnings to become errors, with a few exceptions.
    # This list should be considered alongside
    # setup.cfg -> [tool:pytest] -> filterwarnings

    # remove tweaks from other module imports or example runs
    warnings.resetwarnings()
    # restrict
    warnings.filterwarnings("error")
    # allow these, but show them
    warnings.filterwarnings("always", '.*non-standard config type: "foo".*')
    warnings.filterwarnings("always", '.*config type: "MNEE_USE_CUUDAA".*')
    warnings.filterwarnings("always", ".*cannot make axes width small.*")
    warnings.filterwarnings("always", ".*Axes that are not compatible.*")
    warnings.filterwarnings("always", ".*FastICA did not converge.*")
    # ECoG BIDS spec violations:
    warnings.filterwarnings("always", ".*Fiducial point nasion not found.*")
    warnings.filterwarnings("always", ".*DigMontage is only a subset of.*")
    warnings.filterwarnings(  # xhemi morph (should probably update sample)
        "always", ".*does not exist, creating it and saving it.*"
    )
    # internal warnings
    warnings.filterwarnings("default", module="sphinx")
    # allow these warnings, but don't show them
    for key in (
        "invalid version and will not be supported",  # pyxdf
        "distutils Version classes are deprecated",  # seaborn and neo
        "is_categorical_dtype is deprecated",  # seaborn
        "`np.object` is a deprecated alias for the builtin `object`",  # pyxdf
        # nilearn, should be fixed in > 0.9.1
        "In future, it will be an error for 'np.bool_' scalars to",
        # sklearn hasn't updated to SciPy's sym_pos dep
        "The 'sym_pos' keyword is deprecated",
        # numba
        "`np.MachAr` is deprecated",
        # joblib hasn't updated to avoid distutils
        "distutils package is deprecated",
        # jupyter
        "Jupyter is migrating its paths to use standard",
        r"Widget\..* is deprecated\.",
        # PyQt6
        "Enum value .* is marked as deprecated",
        # matplotlib PDF output
        "The py23 module has been deprecated",
        # pkg_resources
        "Implementing implicit namespace packages",
        "Deprecated call to `pkg_resources",
        # nilearn
        "pkg_resources is deprecated as an API",
        r"The .* was deprecated in Matplotlib 3\.7",
        # scipy
        r"scipy.signal.morlet2 is deprecated in SciPy 1\.12",
    ):
        warnings.filterwarnings(  # deal with other modules having bad imports
            "ignore", message=".*%s.*" % key, category=DeprecationWarning
        )
    warnings.filterwarnings(
        "ignore",
        message="Matplotlib is currently using agg, which is a non-GUI backend.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*is non-interactive, and thus cannot.*",
    )
    # seaborn
    warnings.filterwarnings(
        "ignore",
        message="The figure layout has changed to tight",
        category=UserWarning,
    )
    # matplotlib 3.6 in nilearn and pyvista
    warnings.filterwarnings("ignore", message=".*cmap function will be deprecated.*")
    # xarray/netcdf4
    warnings.filterwarnings(
        "ignore",
        message=r"numpy\.ndarray size changed, may indicate.*",
        category=RuntimeWarning,
    )
    # qdarkstyle
    warnings.filterwarnings(
        "ignore",
        message=r".*Setting theme=.*6 in qdarkstyle.*",
        category=RuntimeWarning,
    )
    # pandas, via seaborn (examples/time_frequency/time_frequency_erds.py)
    for message in (
        "use_inf_as_na option is deprecated.*",
        r"iteritems is deprecated.*Use \.items instead\.",
        "is_categorical_dtype is deprecated.*",
        "The default of observed=False.*",
    ):
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=FutureWarning,
        )
    # pandas in 50_epochs_to_data_frame.py
    warnings.filterwarnings(
        "ignore", message=r"invalid value encountered in cast", category=RuntimeWarning
    )
    # xarray _SixMetaPathImporter (?)
    warnings.filterwarnings(
        "ignore", message=r"falling back to find_module", category=ImportWarning
    )
    # Sphinx deps
    warnings.filterwarnings(
        "ignore", message="The str interface for _CascadingStyleSheet.*"
    )
    # mne-qt-browser until > 0.5.2 released
    warnings.filterwarnings(
        "ignore",
        r"mne\.io\.pick.channel_indices_by_type is deprecated.*",
    )

    # In case we use np.set_printoptions in any tutorials, we only
    # want it to affect those:
    np.set_printoptions(**_np_print_defaults)


reset_warnings(None, None)


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

# -- Sphinx-gallery configuration --------------------------------------------

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
    'doc_module': ('sphinx_gallery', 'ieeg'),
    "reset_modules": ("matplotlib", Resetter()),  # called w/each script
    "reset_modules_order": "both",
    "show_memory": not sys.platform.startswith(("win", "darwin")),
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

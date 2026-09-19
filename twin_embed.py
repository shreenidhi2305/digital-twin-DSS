"""
twin_embed.py
==============
Lets the unified DSS app open each digital twin's OWN dashboard as a page,
so the whole project runs from a single command:

    streamlit run dashboard.py

Nothing inside the twin folders is rewritten. Each twin's original Streamlit
app (healthcare_twin/app.py, retail_twin/app.py, ...) is executed as-is,
inside a small sandbox that:

  1. puts that twin's folder on sys.path and makes it the working directory
     (the retail twin loads "models/demand_model.pkl" by relative path),
  2. clears any same-named local modules left over from another twin
     (healthcare and retail both have a forecasting.py, for example) before
     AND after the run, so twins never see each other's code,
  3. turns the twin's own st.set_page_config() call into a no-op, because the
     entry point (dashboard.py) already set the page config and Streamlit only
     allows one.

To add a twin (e.g. manufacturing): add one entry to TWINS below. That is all.
"""

import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

# key -> how the twin appears in the app.
#   dir   : folder next to this file
#   entry : that twin's own Streamlit script inside the folder
TWINS = {
    "healthcare": {
        "label": "Healthcare Twin",
        "icon": ":material/local_hospital:",
        "dir": "healthcare_twin",
        "entry": "app.py",
        "blurb": (
            "Hospital capacity and occupancy simulation with Low / Baseline / "
            "High-load scenarios, forecasting, live playback and DSS evidence export."
        ),
    },
    "retail": {
        "label": "Retail Twin",
        "icon": ":material/storefront:",
        "dir": "retail_twin",
        "entry": "app.py",
        "blurb": (
            "Store-network demand and inventory simulation driven by an XGBoost "
            "demand model, with supply-disruption scenarios and playback."
        ),
    },
    # Uncomment once the manufacturing twin's dashboard is final:
    # "manufacturing": {
    #     "label": "Manufacturing Twin",
    #     "icon": ":material/precision_manufacturing:",
    #     "dir": "manufacturing_twin",
    #     "entry": "dashboard_standalone.py",
    #     "blurb": "Machine health monitoring with XGBoost failure prediction on a replayed sensor stream.",
    # },
}


def _purge_twin_modules(twin_dir: Path) -> None:
    """Drop every imported module whose source file lives inside twin_dir."""
    root = str(twin_dir)
    for name, mod in list(sys.modules.items()):
        src = getattr(mod, "__file__", None)
        if src and str(src).startswith(root):
            sys.modules.pop(name, None)


@contextmanager
def _twin_environment(twin_dir: Path):
    prev_cwd = os.getcwd()
    orig_set_page_config = st.set_page_config

    _purge_twin_modules(twin_dir)
    sys.path.insert(0, str(twin_dir))
    os.chdir(twin_dir)
    st.set_page_config = lambda *args, **kwargs: None
    try:
        yield
    finally:
        st.set_page_config = orig_set_page_config
        os.chdir(prev_cwd)
        if str(twin_dir) in sys.path:
            sys.path.remove(str(twin_dir))
        _purge_twin_modules(twin_dir)


def run_twin(key: str) -> None:
    """Run one twin's own dashboard inside the current Streamlit page."""
    spec = TWINS[key]
    twin_dir = HERE / spec["dir"]
    entry = twin_dir / spec["entry"]
    if not entry.exists():
        st.error(f"{spec['label']}: expected {entry} but it does not exist. "
                 f"Check the 'dir' / 'entry' values for '{key}' in twin_embed.TWINS.")
        return
    with _twin_environment(twin_dir):
        runpy.run_path(str(entry), run_name="__main__")


def _make_page_fn(key: str):
    def page():
        run_twin(key)
    page.__name__ = f"{key}_twin"
    return page


def build_twin_pages() -> dict:
    """One st.Page per registered twin, keyed like TWINS."""
    return {
        key: st.Page(
            _make_page_fn(key),
            title=spec["label"],
            icon=spec["icon"],
            url_path=f"{key}-twin",
        )
        for key, spec in TWINS.items()
    }

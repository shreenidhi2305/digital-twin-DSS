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
  2. clears same-named modules cached from another twin
     (healthcare and retail both have a forecasting.py, for example) before
     the run, so twins never see each other's code,
  3. turns the twin's own st.set_page_config() call into a no-op, because the
     entry point (dashboard.py) already set the page config and Streamlit only
     allows one.

To add another twin: add one entry to TWINS below. That is all.
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
    "manufacturing": {
        "label": "Manufacturing Twin",
        "icon": ":material/precision_manufacturing:",
        "dir": "manufacturing_twin",
        "entry": "dashboard_standalone.py",
        "blurb": (
            "Machine health monitoring: XGBoost failure prediction on a replayed "
            "sensor stream, with health status, recommendation and DSS evidence packet."
        ),
    },
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
}


def _purge_foreign_modules(twin_dir: Path) -> None:
    """
    Drop cached modules that share a filename with one of this twin's modules
    but were loaded from a DIFFERENT folder (e.g. healthcare's forecasting.py
    when the retail twin is about to import its own). This twin's own modules
    are left alone, so their classes stay the same objects across reruns
    (st.cache_data pickles them by module name and needs that).
    """
    root = os.path.normcase(os.path.abspath(twin_dir))
    for name in {p.stem for p in twin_dir.glob("*.py")}:
        mod = sys.modules.get(name)
        src = getattr(mod, "__file__", None)
        if mod is not None and not (src and os.path.normcase(os.path.abspath(src)).startswith(root)):
            sys.modules.pop(name, None)


@contextmanager
def _twin_environment(twin_dir: Path):
    prev_cwd = os.getcwd()
    orig_set_page_config = st.set_page_config

    _purge_foreign_modules(twin_dir)
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


# Session-state keys that more than one twin app uses for the same purpose
# (retail and manufacturing both keep a "playing" flag). Streamlit shares
# st.session_state across pages, so without this, pressing Play in one twin
# would leave the other twin auto-playing when you open it.
_SHARED_STATE_KEYS = ("playing",)


def _reset_shared_state_on_switch(key: str) -> None:
    if st.session_state.get("_dss_active_twin") != key:
        for k in _SHARED_STATE_KEYS:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["_dss_active_twin"] = key


def run_twin(key: str) -> None:
    """Run one twin's own dashboard inside the current Streamlit page."""
    spec = TWINS[key]
    twin_dir = HERE / spec["dir"]
    entry = twin_dir / spec["entry"]
    if not entry.exists():
        st.error(f"{spec['label']}: expected {entry} but it does not exist. "
                 f"Check the 'dir' / 'entry' values for '{key}' in twin_embed.TWINS.")
        return
    _reset_shared_state_on_switch(key)
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

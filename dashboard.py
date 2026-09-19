"""
dashboard.py
=============
Entry point for the whole DSS project:

    streamlit run dashboard.py

Pages
-----
  DSS Overview      dss_overview.py  -- the fused NLP + twin Readiness Index
                                        (this used to be all of dashboard.py)
  <one per twin>    twin_embed.py    -- each digital twin's own dashboard

To add the manufacturing twin later, see the TWINS registry in twin_embed.py.
"""

import streamlit as st

from twin_embed import build_twin_pages

# Streamlit allows one set_page_config per run, so it lives here. The twins'
# own set_page_config calls are neutralised in twin_embed.py.
st.set_page_config(page_title="Digital Twin DSS", page_icon="\U0001F9ED", layout="wide")

overview = st.Page(
    "dss_overview.py", title="DSS Overview", icon=":material/dashboard:",
    url_path="overview", default=True,
)
twin_pages = build_twin_pages()

# The overview page reads these to render its "open a twin" buttons.
st.session_state["_dss_twin_pages"] = twin_pages

pages = [overview, *twin_pages.values()]
try:
    nav = st.navigation(pages, position="top")   # keeps the sidebar free for each twin's controls
except TypeError:                                # older Streamlit without position=
    nav = st.navigation(pages)
nav.run()

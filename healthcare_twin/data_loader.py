"""
data_loader.py
===============
Thin backward-compatible wrapper over data_connector.py.

The actual data-connection logic (and the honest "batch replay, not live"
disclosure) lives in data_connector.py now. These two module-level
functions are kept so the rest of the pipeline can keep calling
data_loader.load_pmc_infrastructure() / load_operational_stream()
unchanged -- only the connector underneath is swappable.
"""

from data_connector import get_default_connector

_connector = get_default_connector()


def load_pmc_infrastructure():
    """Load the PMC hospital infrastructure dataset (static facility profile)."""
    return _connector.load_pmc_infrastructure()


def load_operational_stream():
    """
    Load dataset3: the external hourly operational/demand time series.

    IMPORTANT: this is NOT the historical record of any specific PMC
    hospital. It supplies a temporal demand pattern only.
    """
    return _connector.load_operational_stream()

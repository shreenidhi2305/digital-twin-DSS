"""
data_connector.py
===================
Explicit, swappable data-connection layer for the Digital Twin.

Per Kritzinger et al. (2018)'s Digital Model / Digital Shadow / Digital
Twin maturity taxonomy, what distinguishes these levels is whether data
flows between the physical and virtual object are AUTOMATED or MANUAL, in
each direction. This prototype is a Digital Model: the connection to data
sources is architected here as a swappable interface, but the concrete
implementation actually used is a one-off/batch CSV replay, not an
automated live feed -- there is no real-time hospital telemetry available
for this project.

Making the connector an explicit interface (rather than bare
pd.read_csv calls scattered through the app) is what lets that scoping
decision be DEMONSTRATED in code, not just asserted in the report: swap
CSVReplayConnector for a real live implementation and nothing else in the
pipeline needs to change (hospital_selector.py, demand_intensity.py,
digital_twin.py, etc. all consume the returned DataFrames the same way
regardless of where they came from).
"""

from abc import ABC, abstractmethod

import pandas as pd

import config


class DataConnector(ABC):
    """Interface every data source for the twin must implement."""

    #: Human-readable description surfaced in the UI, so the connector
    #: actually in use is never hidden from whoever is reading the
    #: twin's output.
    source_type: str = "unspecified"

    #: Whether this connector delivers automated, continuously-updating
    #: data (True) or a static/batch snapshot (False). This is exactly
    #: the Kritzinger et al. distinction between Digital Shadow/Twin
    #: (automated physical -> virtual flow) and Digital Model (manual).
    is_live: bool = False

    @abstractmethod
    def load_pmc_infrastructure(self) -> pd.DataFrame:
        """Return the static facility profile dataset (beds, staff, ...)."""
        raise NotImplementedError

    @abstractmethod
    def load_operational_stream(self) -> pd.DataFrame:
        """Return the hourly operational/demand time series."""
        raise NotImplementedError


class CSVReplayConnector(DataConnector):
    """
    CONCRETE, IN-USE connector. Reads the two static CSVs shipped with the
    project. This is a one-off batch load, replaying dataset3's
    operational stream as a fixed historical file -- not a subscription
    or poll of anything. This is the connector this project actually runs
    on; every other module in the pipeline is written against the
    DataConnector interface, not against this class directly.
    """

    source_type = "Batch CSV replay (offline, one-off load)"
    is_live = False

    def load_pmc_infrastructure(self) -> pd.DataFrame:
        df = pd.read_csv(config.PMC_DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        return df

    def load_operational_stream(self) -> pd.DataFrame:
        """
        IMPORTANT: this is NOT the historical record of any specific PMC
        hospital. It supplies a temporal demand pattern only.
        """
        df = pd.read_csv(config.DATASET3_PATH, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


class LiveFeedConnector(DataConnector):
    """
    DOCUMENTED STUB -- deliberately NOT implemented, and not claimed to
    be. This sketches what an automated physical -> virtual data flow
    would require, so the architecture is demonstrably ready to move from
    Digital Model to Digital Shadow given real infrastructure access,
    without pretending that access exists in this project.

    A real implementation would need, at minimum:
      - Facility profile: a periodic pull from the hospital's own HMIS/HR
        systems (bed counts, staffing rosters) -- e.g. an HL7 FHIR
        Location / PractitionerRole query, or a scheduled export from
        whatever system a facility's infrastructure records are kept in.
      - Operational stream: a live ADT (Admit-Discharge-Transfer) feed --
        HL7 v2 ADT messages or FHIR Encounter resources -- subscribed to
        (webhook / message queue) or polled on an interval, replacing the
        twin's current one-off pandas.read_csv load with a running feed.
      - Authentication against the hospital's system (OAuth2 / mTLS,
        depending on the integration), plus a defined refresh interval.
      - Handling for partial, out-of-order, or duplicate messages, which
        a static CSV never has to deal with but a live ADT feed always
        does.

    None of this exists for this project -- no hospital granted live
    system access to an undergraduate FYP, which is a normal data
    constraint, not a flaw to paper over. Calling either method below
    raises NotImplementedError on purpose: this class documents the seam,
    it does not fake the data behind it.
    """

    source_type = "Live hospital feed (NOT IMPLEMENTED — no live data access for this project)"
    is_live = True

    def load_pmc_infrastructure(self) -> pd.DataFrame:
        raise NotImplementedError(
            "LiveFeedConnector is a documented architectural stub, not a working "
            "connector. No live hospital facility feed is available for this "
            "project -- use CSVReplayConnector."
        )

    def load_operational_stream(self) -> pd.DataFrame:
        raise NotImplementedError(
            "LiveFeedConnector is a documented architectural stub, not a working "
            "connector. No live hospital ADT/operational feed is available for "
            "this project -- use CSVReplayConnector."
        )


def get_default_connector() -> DataConnector:
    """The connector this prototype actually runs on."""
    return CSVReplayConnector()

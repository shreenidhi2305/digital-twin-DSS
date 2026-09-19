"""
hospital_profile.py
=====================
Turns a raw PMC dataset row into a clean, typed hospital profile object
used by the rest of the twin.
"""

from dataclasses import dataclass
import pandas as pd


@dataclass
class HospitalProfile:
    name: str
    total_beds: int
    doctors: int
    nurses: int
    midwives: int
    ambulances: int
    monthly_footfall: float
    facility_type: str
    zone: str
    ward: str

    @property
    def total_staff(self) -> int:
        return self.doctors + self.nurses + self.midwives

    def to_dict(self) -> dict:
        return {
            "hospital_name": self.name,
            "total_beds": self.total_beds,
            "doctors": self.doctors,
            "nurses": self.nurses,
            "midwives": self.midwives,
            "ambulances": self.ambulances,
            "monthly_footfall": self.monthly_footfall,
            "facility_type": self.facility_type,
            "zone": self.zone,
            "ward": self.ward,
        }


def build_profile(row: pd.Series) -> HospitalProfile:
    return HospitalProfile(
        name=row["Facility Name"].strip(),
        total_beds=int(row["Number of Beds in facility type"]),
        doctors=int(row["Number of Doctors / Physicians"]),
        nurses=int(row["Number of Nurses"]),
        midwives=int(row["Number of Midwives Professional"]),
        ambulances=int(row["Count of Ambulance"]),
        monthly_footfall=float(row["Average Monthly Patient Footfall"]),
        facility_type=str(row["Type  (Hospital / Nursing Home / Lab)"]).strip(),
        zone=str(row["Zone Name"]).strip(),
        ward=str(row["Ward Name"]).strip(),
    )

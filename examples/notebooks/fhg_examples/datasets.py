"""Dataset specifications for the example notebooks.

Each dataset is described once here: where its raw and processed files live,
what the label means, and how every sensitive attribute is encoded. callmefair
always treats ``attribute == 1`` as the privileged group, so each
:class:`SensitiveAttribute` documents what 1 and 0 stand for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class SensitiveAttribute:
    name: str
    privileged: str     # meaning of value 1
    unprivileged: str   # meaning of value 0
    rationale: str      # why this split is studied

    @property
    def groups(self) -> tuple[list[dict], list[dict]]:
        """(privileged_group, unprivileged_group) in AIF360 format."""
        return [{self.name: 1}], [{self.name: 0}]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    favorable: str                 # meaning of label == 1
    processed_file: str            # relative to DATA_DIR
    raw_file: str                  # relative to RAW_DIR
    source: str                    # where to obtain the raw file
    sensitive: tuple[SensitiveAttribute, ...]

    @property
    def sensitive_names(self) -> list[str]:
        return [a.name for a in self.sensitive]

    def attribute(self, name: str) -> SensitiveAttribute:
        return next(a for a in self.sensitive if a.name == name)


DIABETES = DatasetSpec(
    name="diabetes",
    label="readmitted",
    favorable="readmitted within 30 days (raw value '<30'); a positive prediction is treated as "
              "the favorable outcome, read here as 'flagged for follow-up care/resources'",
    processed_file="diabetes_pre/diabetes.csv",
    raw_file="diabetic_data.csv",
    source="UCI ML Repository #296, 'Diabetes 130-US Hospitals for Years 1999-2008' "
           "(CC BY 4.0): https://archive.ics.uci.edu/dataset/296",
    sensitive=(
        SensitiveAttribute(
            "age", "younger than 30 (raw bins [0-10), [10-20), [20-30))", "30 or older",
            "The reason for treating under-30s as the reference group is not recorded in the "
            "repository. It is a modelling choice that yields a very small group (about 2.5% "
            "of rows); the notebook shows its size and base rate so you can judge it."),
        SensitiveAttribute(
            "gender", "Male", "Female",
            "Sex/gender is a protected attribute in most anti-discrimination frameworks. "
            "Male is encoded as the reference group; the repository does not record why."),
        SensitiveAttribute(
            "race", "Caucasian", "African American, Hispanic, Asian or Other",
            "Racial and ethnic disparities in US healthcare access and quality are well "
            "documented (e.g. Obermeyer et al., Science 2019, on an algorithm that "
            "under-served Black patients). Caucasian is the reference group, the usual "
            "choice in this literature; 'unprivileged' pools every other recorded race."),
    ),
)

_NOT_RECORDED = "Definition not recorded in the available code."

STROKE = DatasetSpec(
    name="stroke",
    label="stroke",
    favorable="had a stroke; a positive prediction is treated as the favorable outcome, "
              "read here as 'flagged for follow-up care/resources'",
    processed_file="stroke_pre/stroke.csv",
    raw_file="healthcare-dataset-stroke-data.csv",
    source="Kaggle, fedesoriano/stroke-prediction-dataset (healthcare-dataset-stroke-data.csv)",
    sensitive=(
        SensitiveAttribute(
            "age", "65 or older", "younger than 65",
            "65 is the conventional older-adult threshold (also US Medicare eligibility age). It "
            "is fixed a priori: earlier versions tuned the cut-off (73) on the outcome labels, "
            "which makes the group definition depend on the very outcome being audited."),
        SensitiveAttribute(
            "gender", "Male", "Female",
            "Protected attribute in most anti-discrimination frameworks; Male is the reference "
            "group by convention. (One 'Other' row was dropped.)"),
        SensitiveAttribute(
            "ever_married", "Yes (ever married)", "No",
            "Marital status is used as a proxy attribute (strongly linked to age); the reason "
            "for this choice is not recorded."),
        SensitiveAttribute(
            "Residence_type", "Urban", "Rural",
            "Urban/rural residence is a commonly studied access-to-care dimension; "
            "the specific reasoning for this study is not recorded."),
    ),
)

WIDS = DatasetSpec(
    name="wids",
    label="DiagPeriodL90D",
    favorable="metastatic cancer diagnosed within 90 days of screening; a positive prediction is "
              "treated as the favorable outcome, read here as 'flagged for follow-up'",
    processed_file="wids_pre/wids.csv",
    raw_file="training.csv",
    source="Kaggle competition widsdatathon2024-challenge1 (training.csv)",
    sensitive=(
        SensitiveAttribute(
            "patient_age", "younger than 29 ('Younger')", "29 or older ('Older')",
            "From the original workshop notebook; the reason for the cut-off of 29 is not recorded."),
        SensitiveAttribute("race", _NOT_RECORDED, _NOT_RECORDED, _NOT_RECORDED),
        SensitiveAttribute("payer", _NOT_RECORDED, _NOT_RECORDED, _NOT_RECORDED),
        SensitiveAttribute("state_privileged", _NOT_RECORDED, _NOT_RECORDED, _NOT_RECORDED),
        SensitiveAttribute("neighborhood_education", _NOT_RECORDED, _NOT_RECORDED, _NOT_RECORDED),
    ),
)

SPECS: dict[str, DatasetSpec] = {s.name: s for s in (DIABETES, STROKE, WIDS)}


def load(name: str) -> pd.DataFrame:
    """Load the processed (model-ready) CSV shipped with the repository."""
    return pd.read_csv(DATA_DIR / SPECS[name].processed_file)


def load_raw(name: str) -> pd.DataFrame:
    """Load the raw CSV from ``examples/data/raw`` (download it first; see README there)."""
    path = RAW_DIR / SPECS[name].raw_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Source: {SPECS[name].source}")
    return pd.read_csv(path)


def group_summary(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Per sensitive attribute and group: size, share of rows and label base rate."""
    rows = []
    for attr in spec.sensitive:
        for value, meaning in ((1, attr.privileged), (0, attr.unprivileged)):
            sub = df[df[attr.name] == value]
            rows.append({"attribute": attr.name, "value": value,
                         "group": ("privileged: " if value else "unprivileged: ") + meaning,
                         "n": len(sub), "share": len(sub) / len(df),
                         "base_rate": sub[spec.label].mean()})
    return pd.DataFrame(rows)

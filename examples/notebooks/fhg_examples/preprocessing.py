"""Raw -> model-ready preprocessing for the example datasets.

One pure function per dataset (``preprocess_<name>(raw) -> DataFrame``). Each
docstring lists every decision. The output reproduces the CSV shipped under
``examples/data/<name>_pre/`` value for value (see :func:`matches_processed`).

callmefair convention: sensitive attributes are binary columns where
``1`` = privileged (reference) group and ``0`` = unprivileged group, and the
label is binary with ``1`` = favorable outcome.
"""

from __future__ import annotations

import pandas as pd

# ----------------------------------------------------------------- diabetes
DIABETES_LABEL = "readmitted"
YOUNG_AGE_BINS = ("[0-10)", "[10-20)", "[20-30)")     # age == 1 (privileged)
DIABETES_NUMERIC = (
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)
LAB_RESULT_CODES = {"Norm": 0, ">200": 1, ">7": 1, ">300": 2, ">8": 2}   # not measured -> 3
LAB_NOT_MEASURED = 3
LAB_TESTS = ("max_glu_serum", "A1Cresult")
MEDICATION_CODES = {"Steady": 0, "Up": 1, "Down": 2, "No": 3}
CONSTANT_MEDICATIONS = ("examide", "citoglipton")                          # always 'No' -> dropped
FIRST_MEDICATION_COL, LAST_MEDICATION_COL = "metformin", "metformin-pioglitazone"
DIAGNOSIS_COLS = ("diag_1", "diag_2", "diag_3")
# ICD-9 chapters as grouped by Strack et al. (2014). The misspelling is kept
# on purpose: it is the spelling of the shipped column names.
ICD9_RANGES = (
    ("Circulatory", 390, 459), ("Respiratory", 460, 519), ("Digestive", 520, 579),
    ("Genitourinary", 580, 629), ("Muscoloskeletal", 710, 739), ("Neoplasms", 140, 239),
    ("Injury", 800, 999),
)
ICD9_EXTRA_CODES = {"Circulatory": 785, "Respiratory": 786, "Digestive": 787, "Genitourinary": 788}


def icd9_group(code: str) -> str:
    """Map an ICD-9 code (e.g. '250.83', 'V57', '?') to a coarse disease group."""
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    if code[0] in "VE?":
        return "Other"
    value = float(code)
    for group, extra in ICD9_EXTRA_CODES.items():
        if int(value) == extra:
            return group
    for group, low, high in ICD9_RANGES:
        if low <= value <= high:
            return group
    return "Other"


def medication_columns(raw: pd.DataFrame) -> list[str]:
    cols = list(raw.loc[:, FIRST_MEDICATION_COL:LAST_MEDICATION_COL].columns)
    return [c for c in cols if c not in CONSTANT_MEDICATIONS]


def preprocess_diabetes(raw: pd.DataFrame) -> pd.DataFrame:
    """UCI 'Diabetes 130-US hospitals' -> model-ready frame (99,492 x 67).

    Decisions
    * Drop rows with unknown race ('?') or gender ('Unknown/Invalid').
    * Label ``readmitted`` = 1 iff readmitted '<30' days (else '>30' or 'NO').
    * ``race`` 1 = Caucasian, 0 = any other; ``gender`` 1 = Male;
      ``age`` 1 = younger than 30 (bins [0-10), [10-20), [20-30)).
    * Lab tests: Norm=0, elevated=1, very elevated=2, not measured=3.
    * 21 medication columns: Steady=0, Up=1, Down=2, No=3.
    * ``change`` 1 = 'Ch'; ``diabetesMed`` 1 = 'Yes'.
    * diag_1..3 -> one-hot ICD-9 disease group.
    * Dropped: encounter_id, patient_nbr (identifiers), weight, payer_code,
      medical_specialty (mostly missing), examide, citoglipton (constant).
    """
    df = raw[(raw.race != "?") & (raw.gender != "Unknown/Invalid")].reset_index(drop=True)
    out = pd.DataFrame({
        "race": (df.race == "Caucasian").astype(int),
        "gender": (df.gender == "Male").astype(int),
        "age": df.age.isin(YOUNG_AGE_BINS).astype(int),
    })
    out[list(DIABETES_NUMERIC)] = df[list(DIABETES_NUMERIC)]
    for col in LAB_TESTS:
        out[col] = df[col].map(LAB_RESULT_CODES).fillna(LAB_NOT_MEASURED).astype(int)
    for col in medication_columns(raw):
        out[col] = df[col].map(MEDICATION_CODES)
    out["change"] = (df.change == "Ch").astype(int)
    out["diabetesMed"] = (df.diabetesMed == "Yes").astype(int)
    out[DIABETES_LABEL] = (df.readmitted == "<30").astype(int)
    dummies = [pd.get_dummies(df[d].map(icd9_group), prefix=d).astype(int) for d in DIAGNOSIS_COLS]
    out = pd.concat([out, *dummies], axis=1)
    return out[[c for c in out.columns if c != DIABETES_LABEL] + [DIABETES_LABEL]]  # label last


# ------------------------------------------------------------------- stroke
STROKE_LABEL = "stroke"
STROKE_AGE_THRESHOLD = 65          # age >= 65 -> 1 (privileged); a-priori, not tuned on labels
AGE_BAND_EDGES = (-float("inf"), 2, 12, 18, 35, 60, float("inf"))     # right-closed intervals
AGE_BAND_NAMES = ("Infant", "Child", "Adolescent", "Young", "Adults", "Old")
STROKE_BINARY_CODES = {
    "gender": {"Female": 0, "Male": 1},
    "ever_married": {"No": 0, "Yes": 1},
    "Residence_type": {"Rural": 0, "Urban": 1},
}
STROKE_ONE_HOT = ("work_type", "smoking_status")


def preprocess_stroke(raw: pd.DataFrame) -> pd.DataFrame:
    """Kaggle stroke data -> model-ready frame (5,109 rows, raw order, no resampling).

    Decisions
    * Drop ``id`` and the single gender 'Other' row.
    * Missing bmi (201 rows) -> mean bmi of the (gender, ever_married, age band) cell. The means
      use all rows, i.e. before any train/test split: a small leak, accepted for simplicity.
      Age bands are contiguous: [0,2], (2,12], (12,18], (18,35], (35,60], >60.
    * Sensitive attributes (1 = privileged/reference): ``age`` 1 = 65 or older (a-priori
      older-adult / US Medicare age; not tuned on the outcome), ``gender`` 1 = Male,
      ``ever_married`` 1 = Yes, ``Residence_type`` 1 = Urban.
    * ``work_type`` and ``smoking_status`` are nominal -> one-hot (all levels kept).
    * No undersampling: the stroke rate stays at the true 4.87%. Handle the imbalance at
      training time instead (e.g. ``BiasSearch(treat_umbalance=True)``).
    * ``stroke`` is the last column.
    """
    df = raw[raw.gender != "Other"].drop(columns="id").reset_index(drop=True)
    band = pd.cut(df.age, AGE_BAND_EDGES, labels=AGE_BAND_NAMES)
    df["bmi"] = df.bmi.fillna(df.groupby([df.gender, df.ever_married, band], observed=True).bmi.transform("mean"))
    df["age"] = (df.age >= STROKE_AGE_THRESHOLD).astype(int)
    for col, codes in STROKE_BINARY_CODES.items():
        df[col] = df[col].map(codes)
    dummies = [pd.get_dummies(df[c], prefix=c).astype(int) for c in STROKE_ONE_HOT]
    df = df.drop(columns=list(STROKE_ONE_HOT))
    df = pd.concat([df, *dummies], axis=1)
    df.columns = [c.lower().replace("-", "_").replace(" ", "_") if c.startswith(STROKE_ONE_HOT) else c for c in df.columns]
    return df[[c for c in df.columns if c != STROKE_LABEL] + [STROKE_LABEL]]


def legacy_stroke_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """The ORIGINAL pipeline's resampling step, kept only to explain why it was replaced.

    70/15/15 stratified split (seed 42); NearMiss undersampling of the 70% train part only;
    the result is train + val + test, which inflates the stroke rate from 4.87% to 13.2% and
    mixes NearMiss-selected rows into validation/test data when the file is re-split later.
    Expects a frame with binary ``age`` and ``stroke`` columns.
    """
    from imblearn.under_sampling import NearMiss
    from sklearn.model_selection import train_test_split

    def split(frame, test_size):
        return train_test_split(frame, test_size=test_size, stratify=frame[["age", STROKE_LABEL]],
                                random_state=42)
    train, rest = split(df, 0.3)
    test, val = split(rest, 0.5)
    X, y = NearMiss().fit_resample(train.drop(columns=STROKE_LABEL), train[STROKE_LABEL])
    return pd.concat([X.assign(**{STROKE_LABEL: y}), val, test], ignore_index=True)


# ----------------------------------------------------------------- wids
WIDS_AGE_THRESHOLD = 29    # patient_age < 29 -> 1 ('Younger', privileged); other WiDS definitions not recorded

# ------------------------------------------------------------------ helpers
PREPROCESSORS = {"diabetes": preprocess_diabetes, "stroke": preprocess_stroke}


def matches_processed(name: str) -> bool:
    """True if preprocessing the raw file reproduces the shipped CSV exactly."""
    from .datasets import load, load_raw

    shipped = load(name)
    rebuilt = PREPROCESSORS[name](load_raw(name))
    if name == "stroke":      # imputed bmi differs ~1e-9 after the CSV round trip
        rebuilt = rebuilt.round(6)
        shipped = shipped.round(6)
    return list(rebuilt.columns) == list(shipped.columns) and rebuilt.equals(shipped)

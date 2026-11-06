# Raw datasets

Raw files are **not** tracked by git (only this README is). Download them here to
run `examples/notebooks/00_data_and_preprocessing.ipynb` end to end and to
re-derive the processed CSVs in `examples/data/*_pre/`. Without them the
notebook falls back to describing the shipped processed CSVs.

| Dataset | Expected file(s) in this folder | Source | Terms |
|---|---|---|---|
| Diabetes 130-US hospitals | `diabetic_data.csv`, `IDS_mapping.csv` | https://archive.ics.uci.edu/dataset/296 (Strack et al., BioMed Research International, 2014) | CC BY 4.0 |
| Stroke prediction | `healthcare-dataset-stroke-data.csv` | Kaggle `fedesoriano/stroke-prediction-dataset` | See the Kaggle dataset page for its license; check before redistributing |
| WiDS Datathon 2024 (Challenge 1) | `training.csv` | Kaggle competition `widsdatathon2024-challenge1` | Competition rules apply; competition data must **not** be redistributed, which is why this folder is gitignored |

## Download

```bash
cd examples/data/raw

# Diabetes (UCI): download and unzip from the URL above, or
curl -L -o diabetes.zip https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
unzip diabetes.zip

# Kaggle (needs the kaggle CLI and an API token; the WiDS competition rules must be accepted on the website first)
kaggle datasets download -d fedesoriano/stroke-prediction-dataset --unzip
kaggle competitions download -c widsdatathon2024-challenge1
unzip widsdatathon2024-challenge1.zip training.csv
```

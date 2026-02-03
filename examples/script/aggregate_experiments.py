import os
import glob
import argparse
from pathlib import Path

import pandas as pd


def _clean_model_name(name: str) -> str:
    """Canonicalize model name by removing noisy object addresses and module prefixes.

    Examples:
    - "<catboost.core.CatBoostClassifier object at 0x1473d0a34320>" -> "CatBoostClassifier"
    - "<aif360.algorithms.inprocessing.adversarial_debiasing.AdversarialDebiasing object at 0x...>" -> "AdversarialDebiasing"
    - "sklearn.linear_model._logistic.LogisticRegression" -> "LogisticRegression"
    - "callmefair.models.tabnet.TabNetClassifier" -> "TabNetClassifier"
    """
    if not isinstance(name, str):
        try:
            name = str(name)
        except Exception:
            return "UnknownModel"

    s = name.strip()
    # Handle angle-bracket object reprs: <module.Class object at 0x...>
    if s.startswith("<") and s.endswith(">"):
        s_inner = s[1:-1]
        # Remove trailing " object at 0x..."
        s_inner = s_inner.split(" object at ")[0]
        # Take last segment after dots as class name
        s = s_inner.split(".")[-1]
    else:
        # If it's a full module path, take the last segment
        if "." in s:
            s = s.split(".")[-1]

    # Some names may include extra qualifiers; ensure alnum/underscore only
    # but keep CamelCase intact
    s = s.replace(" ", "").replace(">", "").replace("<", "")
    return s or "UnknownModel"


def _infer_model_type(clean_name: str) -> str:
    """Infer broad model type from canonical name.

    Returns one of: 'tree_boost', 'tree_bag', 'linear', 'nn', 'tabnet', 'in_processing', 'post_processing', 'other'.
    """
    n = clean_name.lower()
    if n in {"xgbclassifier", "catboostclassifier", "lgbmclassifier"}:
        return "tree_boost"
    if n in {"randomforestclassifier"}:
        return "tree_bag"
    if n in {"logisticregression"}:
        return "linear"
    if n in {"mlpclassifier"}:
        return "nn"
    if n in {"tabnetclassifier"}:
        return "tabnet"
    if n in {"tabtransformer"}:
        return "nn"
    # In-processing recognizers (AIF360 & others)
    if n in {"adversarialdebiasing", "metafairclassifier", "exponentiatedgradient"}:
        return "in_processing"
    # Post-processing recognizers
    if n in {"calibratedequalizedodds", "equalizedodds", "roc"}:
        return "post_processing"
    return "other"


def _canonical_model_key(clean_name: str) -> str:
    """Map cleaned class names to the canonical model keys used by hfgsearch.

    Canonical keys: lr, mlp, xgb, cat, lgbm, tabnet, tabtransformer.
    Unknowns fall back to the lowercase cleaned name.
    """
    n = clean_name.lower()
    mapping = {
        "logisticregression": "lr",
        "mlpclassifier": "mlp",
        "xgbclassifier": "xgb",
        "catboostclassifier": "cat",
        "lgbmclassifier": "lgbm",
        "tabnetclassifier": "tabnet",
        "tabtransformer": "tabtransformer",
    }
    return mapping.get(n, n)


def aggregate_results(
    input_dir: str,
    output_file: str,
    one_baseline_per_model: bool = True,
    include_source: bool = False,
    include_raw_model: bool = False,
) -> None:
    """
    Aggregate multiple experiment CSV files into a single CSV, ensuring only one baseline per model.

    - Reads all `*.csv` files from `input_dir`
    - Concatenates them into a single DataFrame, handling multiple headers properly
    - Optionally annotates rows with the source file path
    - Ensures only one 'baseline' row per `model` is kept (first occurrence)
    - Writes the consolidated result to `output_file` without index
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = glob.glob(str(input_path / "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    frames = []
    expected_columns = None
    total_headers_removed = 0
    
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            
            # Store expected columns from first valid file
            if expected_columns is None:
                expected_columns = list(df.columns)
                print(f"Expected columns: {expected_columns}")
            
            # Filter out rows that are actually headers (when header row appears as data)
            # This happens when multiple processes concatenate files with headers
            # Check if any row contains the exact column names as values
            header_mask = df.apply(lambda row: list(row.astype(str)) == expected_columns, axis=1)
            headers_found = header_mask.sum()
            if headers_found > 0:
                print(f"Removing {headers_found} header row(s) from '{os.path.basename(fp)}'")
                total_headers_removed += headers_found
            
            df_clean = df[~header_mask].copy()
            
            # Skip empty dataframes after header removal
            if df_clean.empty:
                print(f"Skipping '{os.path.basename(fp)}' - no data rows after header removal")
                continue
                
            # Ensure columns match expected structure
            if list(df_clean.columns) != expected_columns:
                print(f"Warning: Column mismatch in '{fp}', skipping file")
                continue
            
            if include_source:
                df_clean["source_file"] = os.path.basename(fp)
                
            frames.append(df_clean)
            
        except Exception as e:
            print(f"Failed to read '{fp}': {e}")

    if not frames:
        print("No valid CSV data loaded.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Sanity: ensure core columns exist
    missing_cols = [c for c in ("model", "BM") if c not in combined.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing in aggregated data: {missing_cols}")

    # Normalize model names as standard behavior; optionally keep raw names
    combined["model_raw"] = combined["model"].astype(str)
    combined["model"] = combined["model_raw"].apply(_clean_model_name)
    combined["model_type"] = combined["model"].apply(_infer_model_type)
    # Add canonical model key column to match hfgsearch model selection
    combined["model_key"] = combined["model"].apply(_canonical_model_key)
    if not include_raw_model:
        combined.drop(columns=["model_raw"], inplace=True)

    if one_baseline_per_model:
        # Keep first baseline per canonical model; drop further baseline duplicates
        mask_baseline = combined["BM"].astype(str).str.lower() == "baseline"
        df_base = combined[mask_baseline]
        df_non = combined[~mask_baseline]
        # Drop duplicates by normalized ('model','BM') keeping the first
        df_base_dedup = df_base.drop_duplicates(subset=["model", "BM"], keep="first")
        combined = pd.concat([df_base_dedup, df_non], ignore_index=True)

    # Optional: remove exact duplicate rows
    combined.drop_duplicates(inplace=True)

    out_path = Path(output_file)
    if out_path.parent:
        os.makedirs(out_path.parent, exist_ok=True)
    combined.to_csv(out_path, index=False)
    
    print(f"Aggregated {len(csv_files)} files into '{out_path}'.")
    print(f"Total header rows removed: {total_headers_removed}")
    print(f"Final data rows: {len(combined)}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_input = str(script_dir / "results")
    default_output = str(script_dir / "results" / "aggregated_results.csv")

    parser = argparse.ArgumentParser(description="Aggregate experiment CSVs, normalize model names, and dedupe baselines")
    parser.add_argument("--input", type=str, default=default_input, help="Input directory containing CSV files")
    parser.add_argument("--output", type=str, default=default_output, help="Output CSV file path")
    parser.add_argument("--keep-all-baselines", action="store_true", help="Keep all baseline rows (default is one per model)")
    parser.add_argument("--include-source", action="store_true", help="Include source file column in output")
    parser.add_argument("--include-raw-model", action="store_true", help="Include original model names in 'model_raw' column")
    # Hidden/advanced: print normalization preview
    parser.add_argument("--print-canonical-preview", action="store_true", help="Print sample of canonical model names and types")
    args = parser.parse_args()

    aggregate_results(
        input_dir=args.input,
        output_file=args.output,
        one_baseline_per_model=not args.keep_all_baselines,
        include_source=args.include_source,
        include_raw_model=args.include_raw_model,
    )

    if args.print_canonical_preview:
        try:
            df = pd.read_csv(args.output)
            cols = [c for c in ["model_raw", "model", "model_key", "model_type", "BM"] if c in df.columns]
            preview = df[cols].head(20)
            print("Canonicalization preview:\n", preview)
        except Exception as e:
            print(f"Failed to load output for preview: {e}")


if __name__ == "__main__":
    main()

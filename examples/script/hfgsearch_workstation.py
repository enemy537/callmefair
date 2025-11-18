import os
import sys
import uuid
import json
from pathlib import Path
import argparse

# Inject repo root for local imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress tqdm globally
os.environ.setdefault("TQDM_DISABLE", "1")

from callmefair.util.fair_util import BMInterface
from callmefair.mitigation.fair_bm import BMType
from callmefair.mitigation.fair_grid import BMGridSearch
from callmefair.mitigation.fair_log import csvLogger, aggregate_csv_files

# Classifiers used in HFGSearch_usage example
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except Exception:
    TabNetClassifier = None

# TabTransformer (TensorFlow / Keras backend, GPU-capable when available)
try:
    from callmefair.util.models import TabTransformer
except Exception:
    TabTransformer = None

from sklearn.base import clone as sklearn_clone


def configure_threads(threads: int):
    """Configure BLAS / OpenMP / Torch threads for single-node execution."""
    threads_str = str(max(1, threads))
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["OPENBLAS_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads_str
    os.environ["NUMEXPR_NUM_THREADS"] = threads_str
    try:
        import torch
        if hasattr(torch, "set_num_threads"):
            torch.set_num_threads(int(threads_str))
    except Exception:
        pass


def build_classifiers(seed: int = 42, threads: int | None = None, use_gpu: bool = True):
    """Build a dictionary of classifiers for workstation runs.

    Parameters
    ----------
    seed : int
        Random seed for all models.
    threads : int | None
        Threads per model; defaults to os.cpu_count().
    use_gpu : bool
        If True, enable GPU-accelerated modes where supported (XGB, CatBoost,
        TabNet, TabTransformer via TensorFlow backend).
    """
    if threads is None:
        threads = max(1, os.cpu_count() or 1)

    # XGBoost GPU options
    xgb_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=threads,
        random_state=seed,
        objective="binary:logistic",
        base_score=0.5,
    )
    if use_gpu:
        # Use GPU histogram and predictor when available
        xgb_params.update(dict(tree_method="gpu_hist", predictor="gpu_predictor"))

    # CatBoost GPU options
    cat_params = dict(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        verbose=False,
        random_state=seed,
        thread_count=threads,
    )
    if use_gpu:
        cat_params.update(dict(task_type="GPU", devices="0"))

    models: dict[str, object] = {
        # Logistic Regression (multithreaded via BLAS/OMP)
        "logreg": LogisticRegression(max_iter=200, solver="saga", random_state=seed),
        # Sklearn MLP as a lightweight baseline (CPU)
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=seed),
        # XGBoost (GPU-accelerated when use_gpu=True)
        "xgb": XGBClassifier(**xgb_params),
        # CatBoost (GPU-accelerated when use_gpu=True)
        "catboost": CatBoostClassifier(**cat_params),
    }

    # Optional TabNet (PyTorch-based, can use GPU)
    if TabNetClassifier is not None:
        if use_gpu:
            models["tabnet"] = TabNetClassifier(seed=seed, device_name="cuda")
        else:
            models["tabnet"] = TabNetClassifier(seed=seed, device_name="cpu")

    # Optional LightGBM (GPU-accelerated when built with GPU support)
    if LGBMClassifier is not None:
        lgbm_params = dict(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=threads,
            random_state=seed,
            verbose=-1,
        )
        if use_gpu:
            # These parameters only take effect if LightGBM was built with GPU support
            lgbm_params.update(dict(device_type="gpu"))
        models["lgbm"] = LGBMClassifier(**lgbm_params)

    # Optional TabTransformer (TensorFlow/Keras backend)
    if TabTransformer is not None:
        # Minimal config; GPU usage is handled internally by TensorFlow
        tab_config = {
            "batch_size": 1024,
            "epochs": 20,
            "learning_rate": 1e-3,
            "verbose": 0,
        }
        models["tabtransformer"] = TabTransformer(config=tab_config, target="target")

    return models


def fresh_model(model):
    """Clone a model to ensure a fresh instance for each run."""
    if model is None:
        return None
    try:
        return sklearn_clone(model)
    except Exception:
        try:
            params = model.get_params()
            return model.__class__(**params)
        except Exception:
            try:
                return model.__class__()
            except Exception:
                return model


def parse_groups(sensitive_cols: list[str]):
    # Default binary mapping: privileged 1, unprivileged 0 for each sensitive
    privileged = [{col: 1} for col in sensitive_cols]
    unprivileged = [{col: 0} for col in sensitive_cols]
    return privileged, unprivileged


def load_dataset(dataset_path: Path):
    import pandas as pd
    df = pd.read_csv(dataset_path)
    return df


def build_bminterface(df, label: str, sensitive: list[str], seed: int = 42):
    """Build BMInterface with minimal label sanitation for common datasets."""
    try:
        import pandas as pd
        from pandas.api.types import is_string_dtype, is_object_dtype, is_numeric_dtype

        ser = df[label]
        # Handle common diabetes readmission labels
        if is_string_dtype(ser) or is_object_dtype(ser):
            up = ser.astype(str).str.upper()
            vals = set(up.unique())
            if {"NO", "<30", ">30"}.intersection(vals):
                df = df.copy()
                df[label] = up.map(lambda v: 0 if v == "NO" else 1).astype(int)
            else:
                uniq = up.unique()
                if len(uniq) == 2:
                    # Deterministic mapping: sort lexicographically to 0/1
                    sorted_vals = sorted(list(set(uniq)))
                    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
                    df = df.copy()
                    df[label] = up.map(mapping).astype(int)
        elif is_numeric_dtype(ser):
            # Normalize numeric binary labels to {0,1} ints if needed
            uniq = pd.unique(ser)
            if len(uniq) == 2:
                vals = sorted(list(set(map(float, uniq))))
                if set(vals) != {0.0, 1.0}:
                    mapping = {vals[0]: 0, vals[1]: 1}
                    df = df.copy()
                    df[label] = ser.map(lambda v: mapping[float(v)]).astype(int)
    except Exception:
        # If any sanitation fails, proceed with original df
        pass

    return BMInterface(
        df=df,
        label=label,
        protected=sensitive,
        random_state=seed,
        test_size=0.1,
        val_size=0.1,
    )


def ensure_balanced_splits(df, label: str, sensitive: list[str], seed: int) -> BMInterface:
    """Ensure train/val/test each contain the same two classes (for TabNet/CatBoost)."""
    max_tries = 15
    last_bmi = None
    for i in range(max_tries):
        bmi = build_bminterface(df, label, sensitive, seed=seed + i)
        last_bmi = bmi
        try:
            import numpy as np
            _, y_tr = bmi.get_train_xy()
            _, y_va = bmi.get_val_xy()
            _, y_te = bmi.get_test_xy()
            s_tr = set(np.unique(y_tr))
            s_va = set(np.unique(y_va))
            s_te = set(np.unique(y_te))
            if len(s_tr) == 2 and s_tr == s_va == s_te:
                return bmi
        except Exception:
            continue
    return last_bmi if last_bmi is not None else build_bminterface(df, label, sensitive, seed=seed)


def get_default_paths():
    rid = uuid.uuid4().hex[:8]
    out_dir = SCRIPT_DIR
    output_json = out_dir / f"hfgsearch_ws_{rid}.json"
    log_file = out_dir / f"hfgsearch_ws_{rid}.log"
    csv_dir = out_dir / "results"
    csv_base = f"experiment_ws_{rid}"
    return output_json, log_file, csv_dir, csv_base


def gather_combinations():
    pre_types = [t for t in BMType if t.is_pre]
    in_types = [t for t in BMType if t.is_in]
    post_types = [t for t in BMType if t.is_pos]
    combos = []
    # Singles
    combos.extend([[p] for p in pre_types])
    combos.extend([[i] for i in in_types])
    combos.extend([[q] for q in post_types])
    # Pairs across categories
    combos.extend([[p, q] for p in pre_types for q in post_types])
    combos.extend([[p, i] for p in pre_types for i in in_types])
    combos.extend([[i, q] for i in in_types for q in post_types])
    # Triples (one from each category)
    combos.extend([[p, i, q] for p in pre_types for i in in_types for q in post_types])
    return combos


def run_grid_for_combo(bmI: BMInterface, model, combo: list, privileged_groups, unprivileged_groups, threads: int | None = None):
    import gc

    gs = BMGridSearch(
        bmI=bmI,
        model=model,
        bm_list=[combo],
        privileged_group=privileged_groups,
        unprivileged_group=unprivileged_groups,
    )
    try:
        if threads and threads > 0:
            try:
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=threads, user_api=["blas", "openmp"]):
                    gs.run_single_sensitive()
            except Exception:
                gs.run_single_sensitive()
        else:
            gs.run_single_sensitive()
    finally:
        # Explicitly drop references and collect to reduce peak memory use
        del gs
        gc.collect()
    return None


def main():
    parser = argparse.ArgumentParser(description="Workstation HFGSearch script (non-MPI)")
    default_dataset = (SCRIPT_DIR.parent / "datasets" / "diabetic_data_cleaned.csv")
    output_json, default_log, csv_dir, csv_base = get_default_paths()

    parser.add_argument("--dataset", type=str, default=str(default_dataset), help="CSV dataset path")
    parser.add_argument("--label", type=str, default="readmitted", help="Label column name")
    parser.add_argument("--sensitive", type=str, nargs="+", default=["age", "gender", "race"], help="Sensitive attribute columns")
    parser.add_argument("--model", type=str, default="logreg", help="Classifier key or 'all'")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of classifier keys to run (space- or comma-separated), or 'all'",
    )
    parser.add_argument("--threads", type=int, default=None, help="Threads to use for CPU-bound models")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage even if available")
    parser.add_argument("--output", type=str, default=str(output_json), help="Output JSON path")
    parser.add_argument("--log", type=str, default=str(default_log), help="Log file path")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Logging
    log_target = Path(args.log)
    if log_target.parent:
        os.makedirs(log_target.parent, exist_ok=True)
    sys.stdout = open(log_target, "w", buffering=1)
    sys.stderr = sys.stdout

    use_gpu = not args.no_gpu

    # Determine threads and configure runtime
    threads = args.threads if args.threads is not None else max(1, os.cpu_count() or 1)
    print(f"[Workstation] Using threads={threads}, GPU={'on' if use_gpu else 'off'}")
    configure_threads(threads)

    # Prepare CSV logger directory
    os.makedirs(csv_dir, exist_ok=True)
    csv_logger = csvLogger(csv_base, path=str(csv_dir))

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)
    bmI = build_bminterface(df, args.label, args.sensitive, seed=args.seed)
    privileged, unprivileged = parse_groups(args.sensitive)

    models = build_classifiers(args.seed, threads=threads, use_gpu=use_gpu)

    # Model selection
    if args.models:
        raw = args.models
        tokens: list[str] = []
        for item in raw:
            tokens.extend(item.split(","))
        selected = [t.strip() for t in tokens if t.strip()]
        if any(s.lower() == "all" for s in selected):
            selected = list(models.keys())
    elif args.model == "all":
        selected = list(models.keys())
    else:
        selected = [args.model]

    invalid = [m for m in selected if m not in models]
    if invalid:
        print(f"Unknown model(s) {invalid}. Available: {list(models.keys())}")
        return

    print(f"Selected models: {selected}")

    combos = gather_combinations()

    tasks = []
    for combo in combos:
        has_in = any(getattr(t, "is_in", False) for t in combo)
        if has_in:
            in_members = [t for t in combo if getattr(t, "is_in", False)]
            in_name = in_members[0].name if in_members else "inProcessing"
            tasks.append((in_name, None, combo))
        else:
            for mk in selected:
                tasks.append((mk, models[mk], combo))

    print(f"Total (model, combo) tasks: {len(tasks)}")

    local_results = []
    import gc
    for idx, (mk, model, combo) in enumerate(tasks):
        try:
            # Use balanced splits for TabNet and CatBoost to avoid single-class issues
            if ((TabNetClassifier is not None and isinstance(model, TabNetClassifier))
                or isinstance(model, CatBoostClassifier)):
                bmI_task = ensure_balanced_splits(df, args.label, args.sensitive, seed=args.seed)
            else:
                bmI_task = bmI

            model_for_run = fresh_model(model) if model is not None else None
            run_grid_for_combo(bmI_task, model_for_run, combo, privileged, unprivileged, threads=threads)

            local_results.append({
                "model": mk,
                "combo": [bt.name if hasattr(bt, "name") else str(bt) for bt in combo],
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
            print(f"Completed {mk} combo {local_results[-1]['combo']}")
        except Exception as e:
            print(f"Error on task ({mk}, {combo}): {e}")
        finally:
            # Release task-scoped heavy objects
            if mk is not None:
                del model_for_run
            if bmI_task is not bmI:
                del bmI_task
            # Periodically trigger garbage collection
            if (idx + 1) % 5 == 0:
                gc.collect()

    # Persist minimal JSON summary
    out_path = Path(args.output)
    if out_path.parent:
        os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(local_results, f, indent=2)
    print(f"Saved results to {out_path}")

    # Aggregate CSVs
    try:
        aggregate_csv_files(
            folder_path=str(csv_dir),
            output_file=str(csv_dir / f"{csv_base}_aggregated.csv"),
            num_processes=max(1, os.cpu_count() or 1),
        )
        print(f"Aggregated CSVs into {csv_dir / f'{csv_base}_aggregated.csv'}")
    except Exception as e:
        print(f"CSV aggregation failed: {e}")


if __name__ == "__main__":
    main()

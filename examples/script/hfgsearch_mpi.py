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
try:
    # TabTransformer (TensorFlow / Keras backend, GPU-capable when available)
    from callmefair.util.models import TabTransformer
except Exception:
    TabTransformer = None
from sklearn.base import clone as sklearn_clone

# MPI
try:
    from mpi4py import MPI
except Exception as e:
    print("mpi4py is required to run this script in parallel.")
    raise


def configure_threads(threads: int):
    threads_str = str(max(1, threads))
    # Force override to ensure cluster defaults don't pin threads to 1
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
    # threadpoolctl enforcement is handled around training calls


def build_classifiers(seed: int = 42, threads: int | None = None):
    # Determine threads per model; default to all cores if not provided
    if threads is None:
        threads = max(1, os.cpu_count() or 1)
    models = {
        # Canonical keys aligned with BaseSearch wrapper_training
        "lr": LogisticRegression(max_iter=200, solver="saga", random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=seed),
        # XGBoost honors n_jobs for tree methods
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=threads,
            random_state=seed,
            objective="binary:logistic",
            base_score=0.5,
        ),
        # CatBoost uses thread_count for CPU parallelism
        "cat": CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=False, random_state=seed, thread_count=threads),
    }
    if TabNetClassifier is not None:
        models["tabnet"] = TabNetClassifier(seed=seed)
    if LGBMClassifier is not None:
        models["lgbm"] = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=threads,
            random_state=seed,
            verbose=-1  # silent mode
        )
    # Optional TabTransformer (TensorFlow/Keras backend, prefers GPU if available)
    if TabTransformer is not None:
        tab_config = {
            "batch_size": 1024,
            "epochs": 20,
            "learning_rate": 1e-3,
            "verbose": 0,
        }
        # Target column will be provided by BMInterface; we use a generic name
        models["tabtransformer"] = TabTransformer(config=tab_config, target="target")
    return models


def fresh_model(model):
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
    # Return a single DataFrame; BMInterface will handle splitting
    df = pd.read_csv(dataset_path)
    return df


def build_bminterface(df, label: str, sensitive: list[str], seed: int = 42):
    """
    Build BMInterface with minimal label sanitation for common datasets.
    - If label contains diabetes-style strings ('NO','<30','>30'), map to binary 0/1.
    - If label is a two-category string/object column, map consistently to 0/1.
    - If label is already numeric binary, normalize to ints.
    """
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
    """
    Ensure train/val/test each contain the same two classes to satisfy TabNet.
    Retries BMInterface splitting with incremented seeds when class sets mismatch.
    """
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
            # try next seed
            continue
    return last_bmi if last_bmi is not None else build_bminterface(df, label, sensitive, seed=seed)


def get_default_paths():
    rid = uuid.uuid4().hex[:8]
    out_dir = SCRIPT_DIR
    output_json = out_dir / f"hfgsearch_{rid}.json"
    log_file = out_dir / f"hfgsearch_{rid}.log"
    csv_dir = out_dir / "results"
    csv_base = f"experiment_{rid}"
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
    gs = BMGridSearch(
        bmI=bmI,
        model=model,
        bm_list=[combo],
        privileged_group=privileged_groups,
        unprivileged_group=unprivileged_groups,
    )
    # Single sensitive workflow used in examples
    if threads and threads > 0:
        try:
            from threadpoolctl import threadpool_limits
            # Enforce both BLAS and OpenMP thread limits during training
            with threadpool_limits(limits=threads, user_api=["blas", "openmp"]):
                gs.run_single_sensitive()
        except Exception:
            gs.run_single_sensitive()
    else:
        gs.run_single_sensitive()
    # Results are written via csvLogger inside BMGridSearch
    return None


def mpi_distribute(items: list, size: int, rank: int):
    return [item for i, item in enumerate(items) if i % size == rank]


def main():
    parser = argparse.ArgumentParser(description="MPI-enabled HFGSearch script")
    default_dataset = (SCRIPT_DIR.parent / "datasets" / "diabetic_data_cleaned.csv")
    output_json, default_log, csv_dir, csv_base = get_default_paths()
    parser.add_argument("--dataset", type=str, default=str(default_dataset), help="CSV dataset path")
    parser.add_argument("--label", type=str, default="readmitted", help="Label column name")
    parser.add_argument("--sensitive", type=str, nargs="+", default=["age", "gender", "race"], help="Sensitive attribute columns")
    parser.add_argument("--model", type=str, default="lr", help="Classifier key or 'all'")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of classifier keys to run (space- or comma-separated), or 'all'. "
            "Canonical keys: lr, mlp, xgb, cat, lgbm, tabnet, tabtransformer. "
            "Aliases accepted: logreg, logistic, catboost, lightgbm, tabtransform"
        ),
    )
    parser.add_argument("--output", type=str, default=str(output_json), help="Output JSON path")
    parser.add_argument("--log", type=str, default=str(default_log), help="Log file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # MPI init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Redirect stdout/stderr per rank
    log_target = Path(args.log)
    if log_target.parent:
        os.makedirs(log_target.parent, exist_ok=True)
    # Append rank id
    rank_log = log_target.with_name(f"{log_target.stem}_r{rank}{log_target.suffix}")
    sys.stdout = open(rank_log, "w", buffering=1)
    sys.stderr = sys.stdout

    print(f"[Rank {rank}] Starting with size={size}")
    # Prepare CSV logger per rank
    os.makedirs(csv_dir, exist_ok=True)
    rank_csv_base = f"{csv_base}_r{rank}"
    csv_logger = csvLogger(rank_csv_base, path=str(csv_dir))

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)
    bmI = build_bminterface(df, args.label, args.sensitive, seed=args.seed)
    privileged, unprivileged = parse_groups(args.sensitive)

    # Determine local ranks per node to allocate threads fairly on each node
    try:
        local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        local_size = max(1, local_comm.Get_size())
        local_rank = local_comm.Get_rank()
    except Exception:
        # Fallback to world values if shared communicator not available
        local_size = max(1, size)
        local_rank = rank
    # Prefer explicit Slurm allocation per task if available
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            total_cpus = max(1, int(slurm_cpus))
        except Exception:
            total_cpus = max(1, os.cpu_count() or 1)
    else:
        total_cpus = max(1, os.cpu_count() or 1)
    base = max(1, total_cpus // local_size)
    remainder = total_cpus % local_size
    threads_per_rank = base + (1 if local_rank < remainder else 0)
    print(f"[Rank {rank}] local_size={local_size} local_rank={local_rank} total_cpus={total_cpus} threads_per_rank={threads_per_rank}")
    configure_threads(threads_per_rank)

    models = build_classifiers(args.seed, threads=threads_per_rank)
    # Alias map so CLI can accept alternative names while canonicalizing to BaseSearch keys
    alias_map = {
        "logreg": "lr",
        "logistic": "lr",
        "catboost": "cat",
        "lightgbm": "lgbm",
        "tabtransform": "tabtransformer",
    }
    if args.models:
        # Normalize tokens: support comma-separated entries and strip whitespace
        raw = args.models
        tokens = []
        for item in raw:
            tokens.extend(item.split(","))
        selected = []
        for t in tokens:
            tok = t.strip()
            if not tok:
                continue
            canon = alias_map.get(tok.lower(), tok.lower())
            selected.append(canon)
        # Special case: allow 'all' via --models
        if any(s.lower() == "all" for s in selected):
            selected = list(models.keys())
    elif args.model == "all":
        selected = list(models.keys())
    else:
        # Canonicalize single --model via alias_map
        selected = [alias_map.get(args.model.lower(), args.model.lower())]
    # Validate selected models
    invalid = [m for m in selected if m not in models]
    if invalid:
        print(f"[Rank {rank}] Unknown model(s) {invalid}. Available: {list(models.keys())}")
        return

    combos = gather_combinations()
    # Create tasks as (model_key, model_instance, combo)
    # If a combo includes in-processing, do NOT pair with external classifiers; use model=None
    tasks = []
    for combo in combos:
        has_in = any(getattr(t, 'is_in', False) for t in combo)
        if has_in:
            in_members = [t for t in combo if getattr(t, 'is_in', False)]
            in_name = in_members[0].name if in_members else 'inProcessing'
            tasks.append((in_name, None, combo))
        else:
            for mk in selected:
                tasks.append((mk, models[mk], combo))
    my_tasks = mpi_distribute(tasks, size, rank)
    print(f"[Rank {rank}] Assigned {len(my_tasks)} (model, combo) tasks")

    local_results = []
    for mk, model, combo in my_tasks:
        try:
            # Use balanced splits for TabNet and CatBoost to avoid single-class or unknown target issues
            if ((TabNetClassifier is not None and isinstance(model, TabNetClassifier))
                or isinstance(model, CatBoostClassifier)):
                bmI_task = ensure_balanced_splits(df, args.label, args.sensitive, seed=args.seed)
            else:
                bmI_task = bmI
            model_for_run = fresh_model(model) if model is not None else None
            res = run_grid_for_combo(bmI_task, model_for_run, combo, privileged, unprivileged, threads=threads_per_rank)
            # After run, BMGridSearch writes CSV via its internal logger.
            # For convenience, also record a minimal JSON entry of the combo and timestamp.
            local_results.append({
                "model": mk,
                "combo": [bt.name if hasattr(bt, "name") else str(bt) for bt in combo],
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
            print(f"[Rank {rank}] Completed {mk} combo {local_results[-1]['combo']}")
        except Exception as e:
            print(f"[Rank {rank}] Error on task ({mk}, {combo}): {e}")

    # Gather from all ranks
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten
        flat = [r for sub in all_results for r in sub]
        out_path = Path(args.output)
        if out_path.parent:
            os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(flat, f, indent=2)
        print(f"[Rank {rank}] Saved results to {out_path}")

        # Aggregate per-rank CSVs into a single file
        try:
            aggregate_csv_files(folder_path=str(csv_dir), output_file=str(csv_dir / f"{csv_base}_aggregated.csv"), num_processes=max(1, os.cpu_count() or 1))
            print(f"[Rank {rank}] Aggregated CSVs into {csv_dir / f'{csv_base}_aggregated.csv'}")
        except Exception as e:
            print(f"[Rank {rank}] CSV aggregation failed: {e}")


if __name__ == "__main__":
    main()

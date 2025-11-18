import os
import argparse
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import uuid
import json
import time
from typing import Dict, List, Tuple, Any

# MPI imports
try:
    from mpi4py import MPI
except ImportError as e:
    print("mpi4py is required to run this script in parallel.")
    raise

# Make local package available without installation
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Disable tqdm progress bars in downstream modules to keep logs clean
os.environ["TQDM_DISABLE"] = "1"

# Core API
from callmefair.search.fair_search import BiasSearch


def configure_threads(threads: int):
    """Configure threading for optimal performance on high-core-count nodes."""
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


def mpi_distribute(items: list, size: int, rank: int) -> list:
    """Distribute items across MPI ranks."""
    items_per_rank = len(items) // size
    remainder = len(items) % size
    
    start_idx = rank * items_per_rank + min(rank, remainder)
    end_idx = start_idx + items_per_rank + (1 if rank < remainder else 0)
    
    return items[start_idx:end_idx]


def create_fairness_visualization(attributes, raw_scores, normalized_scores, title="Attribute Fairness Impact"):
    """
    Create a Plotly visualization showing fairness impact per attribute across models.

    Parameters:
    - attributes: list[str]
      Attribute names corresponding to rows
    - raw_scores: list[list[float]]
      Raw fairness scores per attribute across models
    - normalized_scores: list[list[float]]
      Normalized fairness scores per attribute across models (0..1)
    - title: str
      Figure title

    Returns:
    - fig: plotly.graph_objects.Figure
    """
    # Keep original attribute order for equivalence with notebooks
    sorted_attributes = attributes
    sorted_raw_scores = raw_scores
    sorted_norm_scores = normalized_scores

    # Build figure: per-attribute row with model points colored by normalized fairness
    fig = go.Figure()

    # Color scale for normalized fairness (0=green, 1=red)
    colors = ['green', 'orange', 'red', 'purple', 'blue']
    
    for i, model in enumerate(['Model_' + str(j) for j in range(len(raw_scores[0]))]):
        model_raw = [row[i] for row in sorted_raw_scores]
        model_norm = [row[i] for row in sorted_norm_scores]
        
        fig.add_trace(go.Scatter(
            x=model_raw,
            y=sorted_attributes,
            mode='markers',
            marker=dict(
                size=10,
                color=model_norm,
                colorscale='RdYlGn_r',
                cmin=0,
                cmax=1,
                showscale=True if i == 0 else False,
                colorbar=dict(title="Normalized Fairness") if i == 0 else None
            ),
            name=model,
            text=[f"Raw: {r:.3f}<br>Norm: {n:.3f}" for r, n in zip(model_raw, model_norm)],
            hovertemplate="%{text}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Raw Fairness Score",
        yaxis_title="Attributes",
        width=900,
        height=600,
        showlegend=True
    )
    
    return fig


def run_bias_search_mpi(df: pd.DataFrame, label: str, attributes: list[str], 
                       n_threads: int, models: list[str], iterations: int,
                       comm: MPI.Comm) -> Dict[str, Any]:
    """
    Run bias search with MPI distribution across models.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Distribute models across ranks
    local_models = mpi_distribute(models, size, rank)
    
    if rank == 0:
        print(f"Distributing {len(models)} models across {size} MPI ranks")
        print(f"Models per rank: {[len(mpi_distribute(models, size, r)) for r in range(size)]}")
    
    # Initialize searcher on each rank
    searcher = BiasSearch(df, label, attributes, n_threads=n_threads)
    
    # Evaluate local models
    local_results = {}
    for model in local_models:
        if rank == 0:
            print(f"Rank {rank}: Evaluating model '{model}'...")
        
        start_time = time.time()
        table, printable = searcher.evaluate_average(model_name=model, iterate=iterations)
        end_time = time.time()
        
        local_results[model] = {
            'table': table,
            'printable': printable,
            'time': end_time - start_time
        }
        
        if rank == 0:
            print(f"Rank {rank}: Model '{model}' completed in {end_time - start_time:.2f}s")
    
    # Gather all results to rank 0
    all_results = comm.gather(local_results, root=0)
    
    if rank == 0:
        # Combine results from all ranks
        combined_results = {}
        total_time = 0
        for rank_results in all_results:
            combined_results.update(rank_results)
            for model_data in rank_results.values():
                total_time += model_data['time']
        
        print(f"\nTotal computation time across all ranks: {total_time:.2f}s")
        print(f"Wall clock time saved by parallelization: ~{total_time/size:.2f}s per rank")
        
        # Print results for each model
        for model in models:
            if model in combined_results:
                print(f"\nModel '{model}':")
                print(combined_results[model]['printable'])
        
        return combined_results
    else:
        return {}


def run_combinations_mpi(df: pd.DataFrame, label: str, attributes: list[str],
                        n_threads: int, iterations: int, model: str,
                        comm: MPI.Comm) -> Tuple[List, str]:
    """
    Run combinations analysis with MPI distribution.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"\nRunning 2-way and 3-way combinations analysis with MPI...")
        print(f"Using model: {model}, iterations: {iterations}")
    
    # Initialize searcher on each rank
    searcher = BiasSearch(df, label, attributes, n_threads=n_threads)
    
    # Run combinations on rank 0 (combinations are already optimized internally)
    if rank == 0:
        start_time = time.time()
        comb_table, comb_printable = searcher.evaluate_combinations(
            iterate=iterations, model_name=model
        )
        end_time = time.time()
        
        print(f"Combinations analysis completed in {end_time - start_time:.2f}s")
        print(comb_printable)
        
        return comb_table, comb_printable
    else:
        return [], ""


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 handles argument parsing and prints initial info
    if rank == 0:
        print(f"Starting MPI Bias Search with {size} ranks")
        print(f"Rank 0 handling argument parsing and coordination")
    
    parser = argparse.ArgumentParser(description="MPI-enabled Bias Search Example Script")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to CSV dataset. Defaults to diabetes example in examples/data.")
    parser.add_argument("--label", type=str, default=None,
                        help="Label column name. Defaults to 'readmitted' for diabetes.")
    parser.add_argument("--attributes", type=str, default=None,
                        help="Comma-separated sensitive attributes. Defaults to 'age,gender,race'.")
    parser.add_argument("--models", type=str, default="lr,mlp,xgb,cat,lgbm,tabtransformer",
                        help="Comma-separated models to evaluate (lr, mlp, xgb, cat, lgbm, tabtransformer).")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for evaluation (default: 10).")
    parser.add_argument("--threads_per_rank", type=int, default=None,
                        help="Threads per MPI rank. Auto-calculated if not provided.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save PNG outputs. Defaults to examples/script/")
    parser.add_argument("--combination", type=str, default=None,
                        help="Attribute pair for combination average as 'attr1,attr2' (optional).")
    parser.add_argument("--combination_model", type=str, default="lr",
                        help="Model to use for combination average (default: lr).")
    parser.add_argument("--combinations_model", type=str, default="lr",
                        help="Model to use for evaluate_combinations (default: lr).")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Redirect all prints to a log file path (optional).")

    args = parser.parse_args()

    # Broadcast arguments to all ranks
    args = comm.bcast(args, root=0)

    # Configure threading based on available cores and MPI ranks
    if args.threads_per_rank is None:
        # Auto-calculate threads per rank
        total_cores = os.cpu_count() or 1
        threads_per_rank = max(1, total_cores // size)
        if rank == 0:
            print(f"Auto-detected {total_cores} cores, using {threads_per_rank} threads per rank")
    else:
        threads_per_rank = args.threads_per_rank
        if rank == 0:
            print(f"Using {threads_per_rank} threads per rank (user specified)")
    
    configure_threads(threads_per_rank)

    # Resolve defaults for diabetes example
    examples_dir = str(EXAMPLES_DIR)
    default_dataset = os.path.join(examples_dir, 'data', 'diabetes_pre', 'diabetes.csv')
    dataset = args.dataset or default_dataset
    label = args.label or 'readmitted'
    attributes = args.attributes.split(',') if args.attributes else ['age', 'gender', 'race']
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    iterations = args.iterations

    # Auto-generate output and log names in the script directory if not provided
    rnd = uuid.uuid4().hex[:8]
    default_output_dir = str(SCRIPT_DIR)
    default_log_file = str(SCRIPT_DIR / f"exp_mpi_{rnd}_rank{rank}.log")

    if rank == 0 and not os.path.isfile(dataset):
        raise FileNotFoundError(f"Dataset not found at '{dataset}'. You can provide --dataset PATH.")

    # Optional print redirection to log file (per rank)
    log_target = args.log_file
    if log_target:
        # Add rank to log filename
        log_base, log_ext = os.path.splitext(log_target)
        log_target = f"{log_base}_rank{rank}{log_ext}"
        
        log_dir = os.path.dirname(log_target)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fp = open(log_target, 'w', buffering=1, encoding='utf-8')
        sys.stdout = log_fp
        sys.stderr = log_fp

    # Load dataset on all ranks (could be optimized to broadcast from rank 0)
    if rank == 0:
        print(f"Loading dataset: {dataset}")
    df = pd.read_csv(dataset)

    if rank == 0:
        print(f"Running MPI BiasSearch with:")
        print(f"  Label: '{label}'")
        print(f"  Attributes: {attributes}")
        print(f"  Models: {models}")
        print(f"  Iterations: {iterations}")
        print(f"  Threads per rank: {threads_per_rank}")
        print(f"  Total MPI ranks: {size}")

    # Run bias search with MPI
    start_total = time.time()
    results = run_bias_search_mpi(df, label, attributes, threads_per_rank, models, iterations, comm)
    
    # Run combinations analysis
    comb_table, comb_printable = run_combinations_mpi(
        df, label, attributes, threads_per_rank, iterations, args.combinations_model, comm
    )
    
    end_total = time.time()

    # Only rank 0 handles visualization and file output
    if rank == 0:
        print(f"\nTotal wall clock time: {end_total - start_total:.2f}s")
        
        # Prepare output directory for PNG files
        output_dir = args.output_dir or default_output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving PNG figures to: {output_dir}")

        # Prepare combined visualization inputs
        if results and models:
            base_table = results[models[0]]['table']
            attributes_viz = [row[0] for row in base_table[1:]]

            def extract_scores(model_table):
                return [[row[1], row[2]] for row in model_table[1:]]  # [raw, norm]

            per_model_scores = {m: extract_scores(results[m]['table']) for m in models if m in results}
            raw_scores = [[per_model_scores[m][i][0] for m in models if m in per_model_scores] 
                         for i in range(len(attributes_viz))]
            normalized_scores = [[per_model_scores[m][i][1] for m in models if m in per_model_scores] 
                               for i in range(len(attributes_viz))]

            fig_attr = create_fairness_visualization(
                attributes_viz, raw_scores, normalized_scores,
                title="MPI Classifier Fairness Impact by Attribute"
            )

            # Save attributes figure
            attr_path = os.path.join(output_dir, f"bias_attr_mpi_{rnd}.png")
            fig_attr.write_image(attr_path, format='png', scale=3)
            print(f"Saved: {attr_path}")

        # Build combinations figure if we have results
        if comb_table and len(comb_table) > 1:
            comb_rows = comb_table[1:]
            comb_names = [r[0] for r in comb_rows]
            comb_raw = [r[1] for r in comb_rows]
            comb_norm = [r[2] for r in comb_rows]
            
            fig_combos = go.Figure()
            fig_combos.add_trace(go.Bar(name='Raw', x=comb_names, y=comb_raw))
            fig_combos.add_trace(go.Bar(name='Normalized', x=comb_names, y=comb_norm))
            fig_combos.update_layout(
                barmode='group', 
                title='MPI Combinations Fairness Scores', 
                xaxis_title='Combination', 
                yaxis_title='Score', 
                width=900, 
                height=600
            )
            
            combos_path = os.path.join(output_dir, f"bias_combos_mpi_{rnd}.png")
            fig_combos.write_image(combos_path, format='png', scale=3)
            print(f"Saved: {combos_path}")

        # Optional combination average experiment
        if args.combination:
            try:
                col_1, col_2 = [c.strip() for c in args.combination.split(',')]
                searcher = BiasSearch(df, label, attributes, n_threads=threads_per_rank)
                
                print(f"\nRunning combination average for ({col_1}, {col_2}) with model='{args.combination_model}'")
                comb_avg_table, comb_avg_printable = searcher.evaluate_combination_average(
                    col_1, col_2, iterate=iterations, model_name=args.combination_model
                )
                print(comb_avg_printable)
                
                # Build figure for combination average per operator
                if comb_avg_table and len(comb_avg_table) > 1:
                    ops_rows = comb_avg_table[1:]
                    ops_names = [r[0] for r in ops_rows]
                    ops_raw = [r[2] for r in ops_rows]
                    ops_norm = [r[3] for r in ops_rows]
                    
                    fig_ops = go.Figure()
                    fig_ops.add_trace(go.Bar(name='Raw', x=ops_names, y=ops_raw))
                    fig_ops.add_trace(go.Bar(name='Normalized', x=ops_names, y=ops_norm))
                    fig_ops.update_layout(
                        barmode='group', 
                        title='MPI Set Operation Fairness (Combination Average)', 
                        xaxis_title='Operation', 
                        yaxis_title='Score', 
                        width=900, 
                        height=600
                    )
                    
                    ops_path = os.path.join(output_dir, f"bias_ops_mpi_{rnd}.png")
                    fig_ops.write_image(ops_path, format='png', scale=3)
                    print(f"Saved: {ops_path}")
                    
            except Exception as e:
                print(f"Error in combination average: {e}")

    # Close log file if used
    if log_target:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            log_fp.close()
        except Exception:
            pass

    # Synchronize all ranks before exit
    comm.Barrier()
    
    if rank == 0:
        print(f"\nMPI Bias Search completed successfully on {size} ranks")


if __name__ == "__main__":
    main()
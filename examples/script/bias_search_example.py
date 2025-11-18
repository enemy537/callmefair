import os
import argparse
from random import random
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import uuid

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


# Update the create_fairness_visualization function to match notebook implementation
def create_fairness_visualization(attributes, raw_scores, normalized_scores, title="Attribute Fairness Impact"):
   def create_fairness_visualization(attributes, raw_scores, normalized_scores, title="Attribute Fairness Impact"):
    """
    Create a visualization of fairness scores with actual data values.
    
    Parameters:
    -----------
    attributes : list
        List of attribute names
    raw_scores : list of lists
        Raw fairness scores for each attribute across multiple measurement points
    normalized_scores : list of lists
        Normalized fairness scores corresponding to raw scores
    title : str
        Title for the plot
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Model names
    model_names = ['Logistic Regression', 'MLP', 'XGBoost', 'CatBoost']
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create figure
    fig = go.Figure()
    
    # Calculate average raw scores for sorting
    avg_raw_scores = [sum(scores)/len(scores) for scores in raw_scores]
    
    # Create sorting indices based on average raw scores
    sort_indices = np.argsort(avg_raw_scores)
    
    # Sort attributes and scores
    sorted_attributes = [attributes[i] for i in sort_indices]
    sorted_raw_scores = [raw_scores[i] for i in sort_indices]
    sorted_norm_scores = [normalized_scores[i] for i in sort_indices]
    
    # Set y-axis labels (attributes)
    y_values = sorted_attributes
    
    # Add scatter points for each attribute
    for i, (attr, raw_list, norm_list) in enumerate(zip(sorted_attributes, sorted_raw_scores, sorted_norm_scores)):
        # Use actual data points instead of generated ones
        x_values = raw_list
        color_values = norm_list
        point_count = len(x_values)
        
        # Add individual points for each model with legend
        for j, (x_val, color_val, model_name, model_color) in enumerate(zip(x_values, color_values, model_names, model_colors)):
            fig.add_trace(go.Scatter(
                x=[x_val],
                y=[i],
                mode='markers',
                marker=dict(
                    size=10,
                    color=model_color,
                    line=dict(width=0),
                    opacity=0.7
                ),
                name=model_name,
                legendgroup=model_name,
                showlegend=(i == 0),  # Only show legend for first attribute
                hoverinfo='text',
                text=f"{attr} ({model_name}): {x_val:.3f}"
            ))
        
        # Add a violin plot for density visualization using actual data
        fig.add_trace(go.Violin(
            x=x_values,
            y=[i] * point_count,
            box_visible=False,
            points=False,
            line_color='rgba(0, 0, 0, 0)',
            fillcolor='rgba(180, 180, 180, 0.3)',
            width=0.6,
            side='both',
            orientation='h',
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add a larger point to highlight the mean raw score
        mean_raw = sum(raw_list) / len(raw_list)
        fig.add_trace(go.Scatter(
            x=[mean_raw],
            y=[i],
            mode='markers',
            marker=dict(
                size=14,
                color='black',
                symbol='diamond',
                line=dict(width=1, color='white'),
                opacity=0.6
            ),
            showlegend=False,  # Don't show in legend
            hoverinfo='text',
            text=f"{attr} (mean): {mean_raw:.3f}"
        ))
    
    # Add a dummy trace for the mean marker legend entry
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=14,
            color='black',
            symbol='diamond',
            line=dict(width=1, color='white'),
            opacity=1.0
        ),
        name='<b>─────────</b><br>mean across models',
        showlegend=True,
        hoverinfo='none'
    ))
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Fairness Score (impact on model output)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            showline=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[min([min(scores) for scores in sorted_raw_scores]) - 0.2, 
                  max([max(scores) for scores in sorted_raw_scores]) + 0.2]
        ),
        yaxis=dict(
            title='',
            tickvals=list(range(len(y_values))),
            ticktext=y_values,
            showline=True,
            linewidth=1,
            linecolor='black',
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        height=max(400, len(sorted_attributes) * 40),
        width=900,
        plot_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=150),
        legend=dict(
            title="Models",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

# Replace the combination bar chart creation with notebook implementation
def process_array(data, group_color):
    """Process 2D array into plot components with border coloring"""
    attributes = [d[0] for d in data[1:]]
    raw_scores = [d[1] for d in data[1:]]
    normalized = [d[2] for d in data[1:]]
    
    # Determine border colors based on normalized scores
    border_colors = ['red' if n == 1 else 'green' for n in normalized]
    
    return {
        'attributes': attributes,
        'raw': raw_scores,
        'border_colors': border_colors,
        'color': group_color
    }


def run_bias_search(df: pd.DataFrame, label: str, attributes: list[str], n_threads: int | None, models: list[str], iterations: int | None):
    # Initialize searcher
    searcher = BiasSearch(df, label, attributes, n_threads=n_threads)

    # Evaluate each requested model
    results = {}
    for model in models:
        table, printable = searcher.evaluate_average(model_name=model, iterate=iterations)
        results[model] = (table, printable)
        # Print concise summary to console
        print(f"\nModel '{model}':")
        print(printable)

    # Prepare combined visualization inputs
    # Assumes tables are lists with header at index 0 and rows as [attr, raw, norm]
    base_table = results[models[0]][0]
    attributes = [row[0] for row in base_table[1:]]

    def extract_scores(model_table):
        return [[row[1], row[2]] for row in model_table[1:]]  # [raw, norm]

    per_model_scores = {m: extract_scores(results[m][0]) for m in models}
    raw_scores = [[per_model_scores[m][i][0] for m in models] for i in range(len(attributes))]
    normalized_scores = [[per_model_scores[m][i][1] for m in models] for i in range(len(attributes))]

    fig = create_fairness_visualization(attributes, raw_scores, normalized_scores,
                                        title="Classifier Fairness Impact by Attribute")
    return fig, results


def run_combination_average(searcher: BiasSearch, col_1: str, col_2: str, iterations: int | None, model: str):
    print(f"\nRunning combination average for ({col_1}, {col_2}) with model='{model}' and iterations={iterations}")
    table, printable = searcher.evaluate_combination_average(col_1, col_2, iterate=iterations, model_name=model)
    print(printable)
    return table, printable


def main():
    parser = argparse.ArgumentParser(description="Bias Search Example Script (converted from notebook)")
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
    parser.add_argument("--n_threads", type=int, default=16,
                        help="Parallel workers for training (default: 16).")
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

    # Resolve defaults for diabetes example
    examples_dir = str(EXAMPLES_DIR)
    default_dataset = os.path.join(examples_dir, 'data', 'diabetes_pre', 'diabetes.csv')
    dataset = args.dataset or default_dataset
    label = args.label or 'readmitted'
    attributes = args.attributes.split(',') if args.attributes else ['age', 'gender', 'race']
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    iterations = args.iterations
    n_threads = args.n_threads

    # Auto-generate output and log names in the script directory if not provided
    rnd = uuid.uuid4().hex[:8]
    default_output_dir = str(SCRIPT_DIR)
    default_log_file = str(SCRIPT_DIR / f"exp_{rnd}.log")

    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"Dataset not found at '{dataset}'. You can provide --dataset PATH.")

    # Optional print redirection to log file
    log_target = args.log_file or default_log_file
    if log_target:
        log_dir = os.path.dirname(log_target)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_fp = open(log_target, 'w', buffering=1, encoding='utf-8')
        sys.stdout = log_fp
        sys.stderr = log_fp

    print(f"Loading dataset: {dataset}")
    df = pd.read_csv(dataset)

    print(f"Running BiasSearch with label='{label}', attributes={attributes}, models={models}, iterations={iterations}, n_threads={n_threads}")
    fig_attr, results = run_bias_search(df, label, attributes, n_threads, models, iterations)

    

    # Prepare output directory for PNG files
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving PNG figures to: {output_dir}")

    # Always run combinations (2-way and 3-way) and include in log
    print("\nRunning 2-way and 3-way combinations analysis...")
    dfbias = BiasSearch(df, label, attributes, n_threads=n_threads)
    comb_table, comb_printable = dfbias.evaluate_combinations(iterate=iterations, model_name=args.combinations_model)
    print(comb_printable)
    
    
    # Process both datasets
    attributes_rows  = results[random.choice(results.keys())][0]
    combos_rows = comb_table

    group1 = process_array(attributes_rows, '#1f77b4')  # Blue
    group2 = process_array(combos_rows, '#ff7f0e')  # Orange

    # Calculate x positions with gap between groups
    x1 = list(range(len(group1['attributes'])))
    x2 = [len(group1['attributes']) + 1 + i for i in range(len(group2['attributes']))]

    # Create bar traces with conditional borders
    trace1 = go.Bar(
        x=x1,
        y=group1['raw'],
        name='Single Attributes',
        marker=dict(
            color=group1['color'],
            line=dict(
                color=group1['border_colors'],
                width=3  # Thicker border for visibility
            )
        )
    )

    trace2 = go.Bar(
        x=x2,
        y=group2['raw'],
        name='Combinations',
        marker=dict(
            color=group2['color'],
            line=dict(
                color=group2['border_colors'],
                width=3
            )
        )
    )

    # Create figure
    fig_combos = go.Figure([trace1, trace2])

    # Calculate axis settings
    tickvals = x1 + x2
    ticktext = group1['attributes'] + group2['attributes']
    separator_pos = len(group1['attributes'])  # Position between groups

    # Update layout with visual separation
    fig_combos.update_layout(
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            title='Attributes',
            showgrid=False
        ),
        yaxis=dict(title='Raw Fairness Score'),
        title='Fairness Analysis with Normalization Indicators',
        bargap=0.25,
        shapes=[dict(
            type='line',
            xref='x',
            yref='paper',
            x0=separator_pos,
            y0=0,
            x1=separator_pos,
            y1=1,
            line=dict(color='gray', width=2, dash='dot')
        )]
    )

    # Optional combination average experiment
    comb_avg_table = None
    comb_avg_printable = None
    if args.combination:
        try:
            col_1, col_2 = [c.strip() for c in args.combination.split(',')]
        except Exception:
            print("--combination expects format 'attr1,attr2'. Skipping.")
            col_1 = col_2 = None

        if col_1 and col_2:
            searcher = BiasSearch(df, label, attributes, n_threads=n_threads)
            comb_avg_table, comb_avg_printable = run_combination_average(searcher, col_1, col_2, iterations, args.combination_model)

    # Save PNGs at high resolution (approx. 300 DPI via scale multiplier)
    attr_path = os.path.join(output_dir, f"bias_attr_{rnd}.png")
    combos_path = os.path.join(output_dir, f"bias_combos_{rnd}.png")
    fig_attr.write_image(attr_path, format='png', scale=3)
    fig_combos.write_image(combos_path, format='png', scale=3)
    print(f"Saved: {attr_path}\nSaved: {combos_path}")

    if comb_avg_table is not None:
        # Build figure for combination average per operator
        ops_rows = comb_avg_table[1:]
        ops_names = [r[0] for r in ops_rows]
        ops_raw = [r[2] for r in ops_rows]
        ops_norm = [r[3] for r in ops_rows]
        fig_ops = go.Figure()
        fig_ops.add_trace(go.Bar(name='Raw', x=ops_names, y=ops_raw))
        fig_ops.add_trace(go.Bar(name='Normalized', x=ops_names, y=ops_norm))
        fig_ops.update_layout(barmode='group', title='Set Operation Fairness (Combination Average)', xaxis_title='Operation', yaxis_title='Score', width=900, height=600)
        ops_path = os.path.join(output_dir, f"bias_ops_{rnd}.png")
        fig_ops.write_image(ops_path, format='png', scale=3)
        print(f"Saved: {ops_path}")

    # Close log file if used
    if log_target:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
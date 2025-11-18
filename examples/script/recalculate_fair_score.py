#!/usr/bin/env python3
"""
Script to recalculate fair_score values in a CSV file.
This script reads fairness metrics from a CSV file and recalculates the fair_score
using the callmefair calculate_fairness_score function.

Usage:
    python recalculate_fair_score.py <input_csv_file> [output_csv_file]
    
If output_csv_file is not provided, the input file will be overwritten.
"""

import sys
import os
import pandas as pd
import argparse
from pathlib import Path

# Add the callmefair package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from callmefair.util.fair_util import calculate_fairness_score
    print("✓ Successfully imported callmefair.util.fair_util.calculate_fairness_score")
except ImportError as e:
    print(f"✗ Failed to import callmefair modules: {e}")
    print("Make sure you're running this script from the callmefair root directory")
    sys.exit(1)

def recalculate_fair_scores(input_file, output_file=None):
    """
    Recalculate fair_score values in a CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to output CSV file. If None, overwrites input file.
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded CSV file: {input_file}")
        print(f"  Shape: {df.shape}")
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        return False
    
    # Check required columns
    required_columns = ['eq_opp_diff', 'avg_odd_diff', 'spd', 'disparate_impact', 'theil_idx']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"✗ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"✓ All required columns found: {required_columns}")
    
    # Display sample of original data
    print("\nSample of original data:")
    print(df[required_columns + (['fair_score'] if 'fair_score' in df.columns else [])].head())
    
    # Recalculate fair_score for each row
    new_fair_scores = []
    calculation_details = []
    
    for idx, row in df.iterrows():
        try:
            # Extract fairness metrics
            EOD = row['eq_opp_diff']
            AOD = row['avg_odd_diff'] 
            SPD = row['spd']
            DI = row['disparate_impact']
            TI = row['theil_idx']
            
            # Calculate fair score
            result = calculate_fairness_score(
                EOD=EOD,
                AOD=AOD, 
                SPD=SPD,
                DI=DI,
                TI=TI
            )
            
            new_fair_score = result['overall_score']
            new_fair_scores.append(new_fair_score)
            
            # Store calculation details for verification
            calculation_details.append({
                'row': idx,
                'EOD': EOD,
                'AOD': AOD,
                'SPD': SPD,
                'DI': DI,
                'TI': TI,
                'new_fair_score': new_fair_score,
                'is_fair': result['is_fair'],
                'old_fair_score': row.get('fair_score', 'N/A')
            })
            
        except Exception as e:
            print(f"✗ Error calculating fair_score for row {idx}: {e}")
            print(f"  Row data: EOD={row.get('eq_opp_diff')}, AOD={row.get('avg_odd_diff')}, "
                  f"SPD={row.get('spd')}, DI={row.get('disparate_impact')}, TI={row.get('theil_idx')}")
            # Use NaN for failed calculations
            new_fair_scores.append(float('nan'))
            calculation_details.append({
                'row': idx,
                'error': str(e),
                'new_fair_score': float('nan')
            })
    
    # Update the dataframe with new fair_score values
    df['fair_score'] = new_fair_scores
    
    # Display sample of updated data
    print(f"\n✓ Recalculated fair_score for {len(df)} rows")
    print("\nSample of updated data:")
    print(df[required_columns + ['fair_score']].head())
    
    # Show statistics
    print(f"\nFair Score Statistics:")
    print(f"  Min: {df['fair_score'].min():.6f}")
    print(f"  Max: {df['fair_score'].max():.6f}")
    print(f"  Mean: {df['fair_score'].mean():.6f}")
    print(f"  Std: {df['fair_score'].std():.6f}")
    
    # Count fair vs unfair
    fair_count = sum(1 for detail in calculation_details 
                    if 'is_fair' in detail and detail['is_fair'])
    unfair_count = sum(1 for detail in calculation_details 
                      if 'is_fair' in detail and not detail['is_fair'])
    
    print(f"  Fair entries: {fair_count}")
    print(f"  Unfair entries: {unfair_count}")
    
    # Show some examples of changes (if original fair_score existed)
    if 'fair_score' in df.columns and len([d for d in calculation_details if d.get('old_fair_score') != 'N/A']) > 0:
        print(f"\nSample of fair_score changes:")
        for i, detail in enumerate(calculation_details[:5]):
            if detail.get('old_fair_score') != 'N/A':
                old_score = detail['old_fair_score']
                new_score = detail['new_fair_score']
                change = new_score - old_score if not pd.isna(old_score) and not pd.isna(new_score) else 'N/A'
                print(f"  Row {detail['row']}: {old_score:.6f} → {new_score:.6f} (Δ: {change})")
    
    # Save the updated CSV
    output_path = output_file if output_file else input_file
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✓ Successfully saved updated CSV to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving CSV file: {e}")
        return False

def main():
    """Main function to handle command line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Recalculate fair_score values in a CSV file using callmefair calculate_fairness_score function"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input CSV file containing fairness metrics"
    )
    parser.add_argument(
        "output_file", 
        nargs='?', 
        default=None,
        help="Path to output CSV file (optional, defaults to overwriting input file)"
    )
    parser.add_argument(
        "--preview", 
        action="store_true",
        help="Preview the recalculation without saving changes"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"✗ Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    print(f"Recalculating fair_score values...")
    print(f"Input file: {args.input_file}")
    
    if args.preview:
        print("Preview mode: No changes will be saved")
        # For preview, we'll use a temporary output to avoid overwriting
        success = recalculate_fair_scores(args.input_file, "/tmp/preview_output.csv")
    else:
        output_file = args.output_file if args.output_file else args.input_file
        print(f"Output file: {output_file}")
        
        if args.output_file is None:
            response = input("This will overwrite the input file. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
        
        success = recalculate_fair_scores(args.input_file, output_file)
    
    if success:
        print("\n✓ Fair score recalculation completed successfully!")
    else:
        print("\n✗ Fair score recalculation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
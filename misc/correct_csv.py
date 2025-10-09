#!/usr/bin/env python3
"""
correct_csv.py - CSV Data Processing Script

This script processes a CSV file containing physics data by:
1. Loading the CSV data
2. Cleaning outliers from specified columns
3. Generating histogram plots of test statistics (observed and Asimov)
4. Saving the cleaned data to an output file

Usage:
    python correct_csv.py input.csv
    python correct_csv.py input.csv -o output.csv
    python correct_csv.py input.csv --output cleaned_data.csv --no-save --threshold 50000
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_inspect_data(input_file):
    """
    Load CSV data and display basic information about the dataset.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Data loaded successfully from: {input_file}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def clean_outliers(df, outlier_threshold=1e5):
    """
    Remove rows with outlier values in specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        outlier_threshold (float): Threshold above which values are considered outliers
        
    Returns:
        pd.DataFrame: Cleaned dataframe with outliers removed
    """
    cols_to_correct = [
        'nLL_exp_mu0', 'nLL_exp_mu1', 'nLL_obs_mu0', 'nLL_obs_mu1',
        'nLLA_exp_mu0', 'nLLA_exp_mu1', 'nLLA_obs_mu0', 'nLLA_obs_mu1'
    ]
    
    # Check which columns actually exist in the dataframe
    existing_cols = [col for col in cols_to_correct if col in df.columns]
    missing_cols = [col for col in cols_to_correct if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: The following expected columns are missing: {missing_cols}")
    
    if not existing_cols:
        print("Warning: None of the expected columns for outlier removal were found.")
        return df
    
    initial_shape = df.shape
    print(f"\nCleaning outliers from columns: {existing_cols}")
    print(f"Outlier threshold: {outlier_threshold}")
    
    # Remove outliers
    for col in existing_cols:
        df = df[~(df[col] > outlier_threshold)]
    
    final_shape = df.shape
    removed_rows = initial_shape[0] - final_shape[0]
    print(f"Removed {removed_rows} rows with outliers")
    print(f"Final dataset shape: {final_shape}")
    
    return df


def create_histogram_plots(df, save_plot=False, output_dir="./"):
    """
    Create and display/save histograms of test statistics for all four combinations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        save_plot (bool): Whether to save the plot to file
        output_dir (str): Directory to save the plot
    """
    # Define the test statistics to plot
    test_stats = [
        {
            'cols': ['nLL_exp_mu0', 'nLL_exp_mu1'],
            'name': 'ΔnLL (Expected)',
            'title': 'Expected mu1 - mu0'
        },
        {
            'cols': ['nLL_obs_mu0', 'nLL_obs_mu1'],
            'name': 'ΔnLL (Observed)',
            'title': 'Observed mu1 - mu0'
        },
        {
            'cols': ['nLLA_exp_mu0', 'nLLA_exp_mu1'],
            'name': 'ΔnLLA (Asimov Expected)',
            'title': 'Asimov Expected mu1 - mu0'
        },
        {
            'cols': ['nLLA_obs_mu0', 'nLLA_obs_mu1'],
            'name': 'ΔnLLA (Asimov Observed)',
            'title': 'Asimov Observed mu1 - mu0'
        }
    ]
    
    # Check which test statistics can be calculated
    available_stats = []
    for stat in test_stats:
        if all(col in df.columns for col in stat['cols']):
            available_stats.append(stat)
        else:
            missing = [col for col in stat['cols'] if col not in df.columns]
            print(f"Warning: Cannot plot {stat['name']}. Missing columns: {missing}")
    
    if not available_stats:
        print("Warning: Cannot create any histograms. Required columns are missing.")
        return
    
    # Create subplots
    n_plots = len(available_stats)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distribution of Test Statistics', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Colors for each plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, stat in enumerate(available_stats):
        try:
            # Calculate test statistic
            test_stat = df[stat['cols'][1]] - df[stat['cols'][0]]
            
            # Create histogram on appropriate subplot
            ax = axes_flat[i]
            n, bins, patches = ax.hist(test_stat, bins=100, alpha=0.7, 
                                     edgecolor='black', linewidth=0.5, 
                                     color=colors[i])
            
            ax.set_xlabel(stat['name'])
            ax.set_ylabel('Counts')
            ax.set_yscale('log')
            ax.set_title(stat['title'])
            ax.grid(True, alpha=0.3)
            
            # Add statistics to the plot
            mean_val = test_stat.mean()
            std_val = test_stat.std()
            median_val = test_stat.median()
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='-.', alpha=0.8, 
                      label=f'Median: {median_val:.2f}')
            
            # Add text box with statistics
            stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nN = {len(test_stat):,}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.8))
            
            ax.legend(loc='upper right')
            
        except Exception as e:
            print(f"Error creating histogram for {stat['name']}: {e}")
            continue
    
    # Hide unused subplots
    for j in range(len(available_stats), 4):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = Path(output_dir) / "test_statistics_histograms.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Histograms saved to: {plot_path}")
    
    plt.show()


def save_cleaned_data(df, output_file):
    """
    Save the cleaned dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe to save
        output_file (str): Path to the output CSV file
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process CSV data by cleaning outliers and generating visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python correct_csv.py data.csv
    python correct_csv.py data.csv -o cleaned_data.csv
    python correct_csv.py data.csv --output cleaned_data.csv --save-plot
    python correct_csv.py data.csv --no-save --threshold 50000
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to output CSV file (default: output.csv in script directory)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not show histogram plots'
    )
    
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save histogram plots to file'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save cleaned data to output file'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=1e5,
        help='Outlier threshold value (default: 100000)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to orchestrate the data processing workflow.
    """
    args = parse_arguments()
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        # Default to output.csv in the same directory as the script
        script_dir = Path(__file__).parent
        output_file = script_dir / "output.csv"
    
    print("=" * 60)
    print("correct_csv.py - CSV Data Processing Script")
    print("=" * 60)
    
    # Load and inspect data
    df = load_and_inspect_data(args.input)
    
    # Clean outliers
    df_cleaned = clean_outliers(df, args.threshold)
    
    # Create histograms
    if not args.no_plot:
        output_dir = Path(output_file).parent
        create_histogram_plots(df_cleaned, args.save_plot, output_dir)
    
    # Save cleaned data
    if not args.no_save:
        save_cleaned_data(df_cleaned, output_file)
    else:
        print("Skipping data save (--no-save flag used)")
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()
import os
import sys
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import json
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def validate_inputs(file_path):
    """Validate the input file path and ensure required files exist."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.endswith('.csv'):
        raise ValueError("Input file must be a CSV.")

    json_file = file_path.replace('csv', 'json')
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"Metadata file not found: {json_file}")

    return json_file

def load_data(file_path):
    """Load data and metadata from files."""
    df = pd.read_csv(file_path)
    with open(file_path.replace('csv', 'json'), 'r') as f:
        metadata = json.load(f)
    return df, metadata

def create_violin_plots(df, labels, output_dir):
    """Generate violin plots for specified columns."""
    for label_set, label_name in labels:
        if not label_set:
            continue

        plt.figure(figsize=(15, 6))
        sns.violinplot(data=df[label_set])
        plt.ylabel('Yields')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'violin_{label_name}.pdf'), dpi=300)
        plt.close()

def create_pairplots(df, lower_limits, upper_limits, observed, labels, output_dir):
    """Create pair plots with overlays for specified data categories."""
    for datasel, datasel_label in labels:
        if not datasel:
            continue

        plt.close()
        nbins = 30
        pgrid = sns.pairplot(df[datasel], corner=True, kind='hist', 
                             plot_kws={'bins': nbins}, diag_kws={'bins': nbins})

        column_indices = [df.columns.get_loc(col) for col in datasel]
        for ii, ii_ind in enumerate(column_indices):
            low_lim_ii = lower_limits[ii_ind]
            up_lim_ii = upper_limits[ii_ind]
            obs = observed[ii_ind][1]
            
            pgrid.axes[ii, ii].axvline(obs, color='orange', linestyle='--')

            for jj, jj_ind in enumerate(column_indices):
                if jj < ii:
                    continue

                pgrid.axes[jj, ii].axvline(up_lim_ii, color='green', linestyle='--')
                pgrid.axes[jj, ii].axvline(low_lim_ii, color='red', linestyle='--')
                if ii != jj:
                    pgrid.axes[jj, ii].scatter(observed[ii_ind][1], 
                                            observed[jj_ind][1], 
                                            color='orange', 
                                            marker='*', s=30)
                    up_lim_jj = upper_limits[jj_ind]
                    low_lim_jj = lower_limits[jj_ind]
                    pgrid.axes[jj, ii].axhline(up_lim_jj, color='green', linestyle='--')
                    pgrid.axes[jj, ii].axhline(low_lim_jj, color='red', linestyle='--')




        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pairplot_{datasel_label}.pdf'), dpi=300)
        plt.close()

def create_histograms(df, metadata, labels, output_dir):
    """Generate histograms for likelihood ratio comparisons."""
    plt.close()
    nbins = 60
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    for i, (label, col_index) in enumerate(labels):
        max_val = np.min(df.iloc[:, col_index])  # Use iloc for positional indexing
        axs[i//2, i%2].hist(2 * (df.iloc[:, col_index + 1] - max_val), bins=nbins, 
                    histtype='stepfilled', alpha=0.7, lw=2, label=label)
        axs[i//2, i%2].set_yscale('log')
        axs[i//2, i%2].set_xlabel(label)
        axs[i//2, i%2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L_ratio_histogram.pdf'), dpi=300)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_scan.py <path_to_csv_file>')
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        json_file = validate_inputs(file_path)
        df, metadata = load_data(file_path)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)

    output_dir = os.path.dirname(file_path)
    columns = df.columns
    # Define signal regions (SRs) and control regions (CRs)
    SR_names, SRs = [], []
    CR_names, CRs = [], []
    with open(json_file, 'r') as metaf:
        metadata = json.load(metaf)

    print(metadata['channels'])
    removed_channels = metadata['remove_channels']
    lower_limits, upper_limits, observed = [], [], []
    ii = 0
    for k, v in metadata['channels'][0].items():
        if k in removed_channels:
            print(f'Skipping removed channel: {k}')
        elif v == 'SR':
            SR_names.append(k)
            lower_limits.append(metadata['lower_limits'][ii])
            upper_limits.append(metadata['upper_limits'][ii])
            observed.append(metadata['obs_yields'][ii])
        elif v == 'CR':
            CR_names.append(k)
            lower_limits.append(metadata['lower_limits'][ii])
            upper_limits.append(metadata['upper_limits'][ii])
            observed.append(metadata['obs_yields'][ii])
        elif v == 'VR':
            pass
        else:
            raise ValueError(f'Unknown channel type: {v}')
        ii += 1

    for col in columns:
        if col.split('-')[0] in SR_names:
            SRs.append(col)
        elif col.split('-')[0] in CR_names:
            CRs.append(col)

    # Violin plots
    create_violin_plots(df, [(SRs, 'SRs'), (CRs, 'CRs')], output_dir)

    # Pair plots
    create_pairplots(df, upper_limits, lower_limits, observed, [(SRs, 'SRs'), (CRs, 'CRs')], output_dir)

    # Histograms
    sr_cr_indices = list(range(len(SRs + CRs)))
    # For debugging, print the indices and check their validity
    print(f"SR/CR Indices: {sr_cr_indices}, DataFrame Columns: {df.columns}")
    x_data_len = len(sr_cr_indices)
    histogram_labels = [
        (r'$-2(\log L_{\mu=1}^{\text{exp}} - \log L_{\text{max}}^{\text{exp}})$', x_data_len),
        (r'$-2(\log L_{\mu=1}^{\text{obs}} - \log L_{\text{max}}^{\text{obs}})$', x_data_len+2),
        (r'$-2(\log L_{\mu=1}^{\text{exp, A}} - \log L_{\text{max}}^{\text{exp, A}})$', x_data_len+4),
        (r'$-2(\log L_{\mu=1}^{\text{obs, A}} - \log L_{\text{max}}^{\text{obs, A}})$', x_data_len+6),
    ]
    create_histograms(df, metadata, histogram_labels, output_dir)

if __name__ == "__main__":
    main()

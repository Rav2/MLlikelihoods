import os
import sys
import logging
from os.path import join, dirname, basename, exists
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import SymmetricalLogLocator
import seaborn as sns
import matplotlib.colors as mcolors
from load_onnx import load_model_and_normalize
from matplotlib.ticker import SymmetricalLogLocator
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

# Set up the logger
module_path = os.path.abspath(os.path.join('..', 'sampling'))
sys.path.append(module_path)
from misc import *
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s  (%(filename)s:%(lineno)d)")
file_handler = logging.FileHandler("log.txt", mode='w')
file_handler.setFormatter(log_formatter)
consoleHandler = logging.StreamHandler()
consoleFormatter = CustomFormatter()
consoleHandler.setFormatter(consoleFormatter)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(consoleHandler)

def create_output_directory(outpath):
    """Create output directory if it does not exist."""
    if not exists(outpath):
        os.mkdir(outpath)
        log.info(f"Created output directory at {outpath}")

def preprocess_data(dataset):
    """Load and preprocess the dataset."""
    df = pd.read_csv(dataset)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset

    # Compute deltas for nLL values
    for i in range(4):
        col_idx = -8 + 2 * i
        df.iloc[:, col_idx + 1] -= df.iloc[:, col_idx]
        df.drop(df.columns[col_idx], axis=1, inplace=True)

    return df

def plot_truth_vs_prediction(ax, y_truth, y_pred, label_x, label_y, title):
    """Helper function to plot truth vs prediction scatter plots."""
    conditions = [
        (np.abs((y_truth - y_pred) / y_pred) <= 0.01),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.10),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.20),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.50),
        (np.abs((y_truth - y_pred) / y_pred) > 0.50),
    ]

    labels = ['APE<=1%', '10%>=APE>1%', '20%>=APE>10%', '50%>=APE>20%', 'APE>50%']
    colors = ['cyan', 'green', 'orange', 'red', 'darkred']

    for i, cond in enumerate(conditions):
        if i == 0 or i == len(conditions)-1:
            filtered_cond = cond
        elif i < len(conditions) - 1:
            filtered_cond = cond & ~conditions[i-1]
        count = np.sum(filtered_cond)
        ax.scatter(y_truth[filtered_cond], y_pred[filtered_cond], marker='.', edgecolor='none', s=20, alpha=0.6, c=colors[i], label=labels[i]+f' ({count})', zorder=1)
        # exit()

    ax.plot([np.min(y_truth), np.max(y_truth)], [np.min(y_truth), np.max(y_truth)], c='black', zorder=0)
    linthresh = 0.01

    ax.set_xscale('symlog', linthresh=linthresh,)
    ax.set_yscale('symlog', linthresh=linthresh,)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend()
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', length=5)
    ax.xaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=linthresh, subs=np.arange(1.0, 10.0) * 0.1))
    ax.yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=linthresh, subs=np.arange(1.0, 10.0) * 0.1))

    ax.grid(which='both', linestyle='--', linewidth=0.25, zorder=0)
    ax.set_title(title)


def plot_with_marginals(y_truth, y_pred, label_x, label_y, title, ax_main, bins=None):
    """
    Plot scatter with stacked marginal histograms around each main axis.
    Assumes ax_main is provided; places horizontal marginal below and vertical to the right.
    Bins match symlog scale if not provided.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import SymmetricalLogLocator

    # Define APE group masks
    conditions = [
        (np.abs((y_truth - y_pred) / y_pred) <= 0.01),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.10),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.20),
        (np.abs((y_truth - y_pred) / y_pred) <= 0.50),
        (np.abs((y_truth - y_pred) / y_pred) > 0.50),
    ]
    colors = ['cyan', 'green', 'orange', 'red', 'darkred']
    labels = [
        f"APE<=1% ({conditions[0].sum()})",
        f"1–10% ({(conditions[1]&~conditions[0]).sum()})",
        f"10–20% ({(conditions[2]&~conditions[1]).sum()})",
        f"20–50% ({(conditions[3]&~conditions[2]).sum()})",
        f">50% ({conditions[4].sum()})"
    ]

    # Get figure and position of main axis
    fig = ax_main.figure
    pos = ax_main.get_position()
    pad = 0.005
    hist_h = 0.12 * pos.height
    hist_w = 0.12 * pos.width

    # horizontal marginal BELOW main
    ax_histx = fig.add_axes([
        pos.x0,
        pos.y0 - pad - hist_h,
        pos.width,
        hist_h
    ])
    # vertical marginal RIGHT of main
    ax_histy = fig.add_axes([
        pos.x1 + pad,
        pos.y0,
        hist_w,
        pos.height
    ])

    # Main scatter
    mn, mx = np.min(y_truth), np.max(y_truth)
    ax_main.plot([mn, mx], [mn, mx], c='black', zorder=0)
    for color, label, mask in zip(colors, labels, [
        conditions[0],
        conditions[1] & ~conditions[0],
        conditions[2] & ~conditions[1],
        conditions[3] & ~conditions[2],
        conditions[4]
    ]):
        ax_main.scatter(
            y_truth[mask], y_pred[mask],
            s=20, alpha=0.6, c=color,
            marker='.', edgecolor='none',
            label=label
        )
    ax_main.tick_params(axis='x', which='both', labelbottom=False)
    # Symlog styling on main
    thresh = 0.01
    ax_main.set_xscale('symlog', linthresh=thresh)
    ax_main.set_yscale('symlog', linthresh=thresh)
    ax_main.set_xlabel(label_x)
    ax_main.set_ylabel(label_y)
    ax_main.set_title(title)
    ax_main.minorticks_on()
    ax_main.tick_params(which='both', direction='in', length=5)
    locator = SymmetricalLogLocator(base=10, linthresh=thresh,
                                    subs=np.arange(1, 10) * 0.1)
    ax_main.xaxis.set_minor_locator(locator)
    ax_main.yaxis.set_minor_locator(locator)
    ax_main.grid(which='both', linestyle='--', linewidth=0.25)
    ax_main.legend(fontsize='small')

    # Compute bins if not provided
    ax_main.relim()
    ax_main.autoscale_view()
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    bins_x = np.unique(locator.tick_values(*xlim))
    bins_y = np.unique(locator.tick_values(*ylim))

    # Align histogram axes limits to main
    ax_histx.set_xscale('symlog', linthresh=thresh)
    ax_histx.set_yscale('log')
    # ax_histx.set_xlim(ax_main.get_xlim())
    ax_histx.minorticks_on()
    ax_histx.hist([
        y_truth[conditions[0]],
        y_truth[conditions[1] & ~conditions[0]],
        y_truth[conditions[2] & ~conditions[1]],
        y_truth[conditions[3] & ~conditions[2]],
        y_truth[conditions[4]]
    ], bins=bins_x, stacked=True, color=colors, alpha=0.5, range=xlim)
    # ax_histx.axis('off')
    ax_histx.set_xlim(xlim)


    ax_histy.set_yscale('symlog', linthresh=thresh)
    ax_histy.set_xscale('log')
    # ax_histy.set_ylim(ax_main.get_ylim())
    ax_histy.minorticks_on()
    ax_histy.hist([
        y_pred[conditions[0]],
        y_pred[conditions[1] & ~conditions[0]],
        y_pred[conditions[2] & ~conditions[1]],
        y_pred[conditions[3] & ~conditions[2]],
        y_pred[conditions[4]]
    ], bins=bins_y, orientation='horizontal', stacked=True, color=colors, alpha=0.5, range=ylim)
    ax_histy.tick_params(axis='y', which='both', labelleft=False)
    ax_histy.set_ylim(ylim)


    return ax_main, ax_histx, ax_histy




def plot_error_histogram(ax, delta_pred, delta_truth, limit, title):
    """Helper function to plot error histograms with log-scale and blue-red colormap."""
    h = ax.hist2d(delta_pred - delta_truth, delta_truth, bins=100, cmap='nipy_spectral_r', norm=LogNorm(), range=[[-1*limit, limit],[np.min(delta_truth), np.max(delta_truth)]])
    ax.set_xlabel('∆_pred - ∆_truth')
    ax.set_ylabel('∆_truth')
    ax.set_xlim(-1*limit, limit)
    ax.set_title(title)
    plt.colorbar(h[3], ax=ax)  # Add colorbar for the 2D histogram


def plot_comparison_histograms(ax, delta_pred, delta_truth, title):
    """
    Helper function to plot a 1D histogram comparing ∆_truth and ∆_pred on a single axis.

    Parameters:
        ax: matplotlib.axes.Axes
            The axis to plot on.
        delta_pred: array-like
            Predicted delta values.
        delta_truth: array-like
            True delta values.
        title: str
            Title for the plot.
    """
    ax.hist(delta_truth, bins=100, alpha=0.3, label='∆_truth', color='blue')
    ax.hist(delta_pred, bins=100, alpha=0.3, label='∆_pred', color='red')
    ax.set_title(title)
    ax.set_xlabel('∆')
    ax.set_yscale('log')
    ax.set_ylabel('Frequency')
    ax.legend()
    

def plot_jointplot_with_ape(y_truth, y_pred, label, outpath, dataset_name):
    """Plot APE jointplot and save it."""
    ape = 100 * np.abs((y_truth - y_pred) / y_truth)
    sns.set(style="white", color_codes=True)
    g = sns.jointplot(x=y_truth, y=ape, bins=50, kind="hist", ratio=3, cmap='plasma',
                      marginal_kws=dict(bins=100, color='blue'))
    log_norm = mcolors.LogNorm(vmin=1, vmax=5e2)
    g.ax_joint.collections[0].set_norm(log_norm)
    plt.colorbar(g.ax_joint.collections[0], ax=g.ax_joint)
    plt.hlines(1.0, np.min(y_truth), np.max(y_truth), linestyle='--', color='gray')
    plt.xlabel(label + ' (truth)')
    plt.ylabel('Absolute Percentage Error [%]')
    plt.tight_layout()
    save_path = join(outpath, f"{dataset_name}-{label}_.pdf")
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved jointplot to {save_path}")

def plot_marginal_histogram(ape, label, outpath, dataset_name):
    """Plot marginal histogram of APE and save it."""
    plt.figure(figsize=(6, 5))
    plt.hist(ape, bins=50, color='blue', log=True)
    plt.xlabel('Absolute Percentage Error [%]')
    plt.tight_layout()
    save_path = join(outpath, f"{dataset_name}-{label}_marginal.pdf")
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved marginal histogram to {save_path}")

def main(model_path, dataset):
    # Set up output directory
    outdir = basename(dirname(dataset))
    outpath = join('../validation', outdir)
    create_output_directory(outpath)
    dataset_name = basename(dataset).replace('.csv', '')
    
    # Preprocess data
    df = preprocess_data(dataset)
    x_test = df.iloc[:, :-4]
    columns = df.columns

    # Load model and normalize data
    sess, x_test_norm, mean_arr, std_arr, *mu0_nLLs = load_model_and_normalize(model_path, df)
    y_test_norm = x_test_norm.iloc[:, -4:] * std_arr[-4:] + mean_arr[-4:]
    x_test_norm = x_test_norm.iloc[:, :-4]

    # Get predictions
    y_pred_norm = sess.run(None, {'input_1': x_test_norm.to_numpy().astype(np.float32)})[0] * std_arr[-4:] + mean_arr[-4:]

    # Check for corrupted points
    for i, label in enumerate(columns[-4:]):
        log.info(f'Checking for high values in {label}')
        corr_cond = y_test_norm.iloc[:, i] > 1e6
        if corr_cond.any():
            log.warning(f"Found {corr_cond.sum()} abnormal points in {label}")

    # Plot Delta truth vs prediction
    fig, axs = plt.subplots(2, 2, figsize=(14, 14), gridspec_kw={'wspace':0.5, 'hspace':0.5})
    plot_with_marginals(y_test_norm.iloc[:, 0], y_pred_norm[:, 0], 'Δ Exp Truth', 'Δ Exp Pred', 'Expected', ax_main=axs[0,0])
    plot_with_marginals(y_test_norm.iloc[:, 1], y_pred_norm[:, 1], 'Δ Obs Truth', 'Δ Obs Pred', 'Observed', ax_main=axs[0,1])
    plot_with_marginals(y_test_norm.iloc[:, 2], y_pred_norm[:, 2], 'Δ Exp Truth', 'Δ Exp Pred', 'Expected Asimov', ax_main=axs[1,0])
    plot_with_marginals(y_test_norm.iloc[:, 3], y_pred_norm[:, 3], 'Δ Obs Truth', 'Δ Obs Pred', 'Observed Asimov', ax_main=axs[1,1])

    # plt.tight_layout()
    plt.savefig(join(outpath, f'{dataset_name}-truth_vs_prediction.pdf'))
    plt.close()

    # Plot error vs delta_truth histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    exp_diff = np.concatenate([ y_test_norm.iloc[:, 0]- y_pred_norm[:, 0], y_test_norm.iloc[:, 2]- y_pred_norm[:, 2] ])
    exp_lim = np.max( [np.abs(np.min(exp_diff)), np.abs(np.max(exp_diff))] ) 
    obs_diff = np.concatenate([ y_test_norm.iloc[:, 1]- y_pred_norm[:, 1], y_test_norm.iloc[:, 3]- y_pred_norm[:, 3] ])
    obs_lim = np.max( [np.abs(np.min(obs_diff)), np.abs(np.max(obs_diff))] )

    plot_error_histogram(axs[0,0], y_pred_norm[:, 0], y_test_norm.iloc[:, 0], exp_lim, 'Expected')
    plot_error_histogram(axs[0,1], y_pred_norm[:, 1], y_test_norm.iloc[:, 1], obs_lim, 'Observed')
    plot_error_histogram(axs[1,0], y_pred_norm[:, 2], y_test_norm.iloc[:, 2], exp_lim, 'Expected Asimov')
    plot_error_histogram(axs[1,1], y_pred_norm[:, 3], y_test_norm.iloc[:, 3], obs_lim, 'Observed Asimov')

    plt.tight_layout()
    plt.savefig(join(outpath, f'{dataset_name}-delta_errors.pdf'))
    plt.close()

    # Plot truth and predictions histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_comparison_histograms(axs[0,0], y_pred_norm[:, 0], y_test_norm.iloc[:, 0], 'Expected')
    plot_comparison_histograms(axs[0,1], y_pred_norm[:, 1], y_test_norm.iloc[:, 1], 'Observed')
    plot_comparison_histograms(axs[1,0], y_pred_norm[:, 2], y_test_norm.iloc[:, 2], 'Expected Asimov')
    plot_comparison_histograms(axs[1,1], y_pred_norm[:, 3], y_test_norm.iloc[:, 3], 'Observed Asimov')

    plt.tight_layout()
    plt.savefig(join(outpath, f'{dataset_name}-delta_hist.pdf'))
    plt.close()

    # Plot APE and histograms for each nLL component
   
    labels = ['nLL_exp_mu1', 'nLL_obs_mu1', 'nLLA_exp_mu1', 'nLLA_obs_mu1']
    for i, label in enumerate(labels):
        l_truth = y_test_norm.iloc[:, i] + mu0_nLLs[i]
        l_pred = y_pred_norm[:, i] + mu0_nLLs[i]
        plot_jointplot_with_ape(l_truth, l_pred, label, outpath, dataset_name)
        ape = 100 * np.abs((l_truth - l_pred) / l_truth)
        plot_marginal_histogram(ape, label, outpath, dataset_name)

    log.info("Processing completed. All plots saved.")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        log.error('Usage: python evaluate.py <path-to-onnx-model> <path-to-test.csv>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

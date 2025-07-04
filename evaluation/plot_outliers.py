#
# author: Rafal Maselek
# e-mail: rafal.maselek@lpsc.in2p3.fr
#
# This script plots distribution of points during the training and position of outliers
#
import os, sys
from os.path import join, basename, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
import json 


def main(data_path, outliers_path, metadata_path):
    # read the csv file
    df_train = pd.read_csv(data_path)
    cols = df_train.columns
    df_train.reset_index(drop=True, inplace=True)
    x_train = df_train.iloc[:,:-4]
    y_train = np.array(df_train.iloc[:, -4:])
    dirn = join('../validation/', basename(dirname(data_path)))
    # read the metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    # make the grid plots
    SRs = 9
    CRs = 5
    outlier_files = ['nLL_exp_mu1_anomalies-train.csv', 'nLL_obs_mu1_anomalies-train.csv', \
                    'nLL_exp_mu1_anomalies-val.csv', 'nLL_obs_mu1_anomalies-val.csv',\
                    'nLL_exp_mu1_anomalies-test.csv', 'nLL_obs_mu1_anomalies-test.csv']
    colors = ['black', 'green', 'red', 'cyan', 'orange', 'magenta']
    
    outlier_data = []
    for ff in range(6):
        try:
            outlier_data.append(pd.read_csv(join(outliers_path, outlier_files[ff]), index_col=0))
        except FileNotFoundError:
            print("[ERROR] File {} doesn't exist!".format(outlier_files[ff]))
            outlier_data.append(None)


    # GRID OF HISTOGRAMS
    nbins = 30
    pgrid = sns.pairplot(df_train.iloc[:, :SRs], corner=True, kind='hist', plot_kws=dict(bins=nbins), diag_kws=dict(bins=nbins))
    p0s = np.array(metadata['starting_points'])
    for ii in range(len(pgrid.axes)):
        low_lim = metadata['lower_limits'][ii]
        up_lim = metadata['upper_limits'][ii]
        init_lim = metadata['initial_lower_limits'][ii]
        chan, obs = metadata['obs_yields'][ii]
        in_the_box_X = (df_train.iloc[:, ii] >= low_lim) & (df_train.iloc[:, ii] <= up_lim)
        pgrid.axes[ii,ii].axvline(obs, color='orange', linestyle='--')
        pgrid.axes[ii,ii].xaxis.set_label_text(df_train.columns[ii], fontsize=10, visible=True)

        for jj in range(ii, len(pgrid.axes)):
            pgrid.axes[jj,ii].axvline(up_lim, color='green', linestyle='--')
            pgrid.axes[jj,ii].axvline(low_lim, color='red', linestyle='--')
            pgrid.axes[jj,ii].axvline(init_lim, color='red', linestyle=':')
            pgrid.axes[jj,ii].set_ylabel(pgrid.axes[jj,ii].get_ylabel(), fontsize=10, visible=True)
            pgrid.axes[jj,ii].xaxis.set_label_text(pgrid.axes[jj,ii].get_xlabel(), fontsize=10, visible=True)
            #pgrid.axes[jj,ii].set_title(f'{jj} {ii}', fontsize=8)
            in_the_box_Y = (df_train.iloc[:, jj] >=  metadata['lower_limits'][jj]) & (df_train.iloc[:, jj] <= metadata['upper_limits'][jj])
            in_the_box_N = np.sum(np.logical_and(in_the_box_X, in_the_box_Y))
            in_the_box_perc = np.round(100.0*in_the_box_N/df_train.iloc[:, jj].count(), 2)
            pgrid.axes[jj,ii].set_title('{:.2f}%'.format(in_the_box_perc), fontsize=8)
            if jj != ii:
                pgrid.axes[jj,ii].scatter(obs, metadata['obs_yields'][jj][1], color='orange', marker='*', zorder=3,)
                for ff in range(6):
                    # read the outliers info
                    df_anom = outlier_data[ff]
                    if df_anom is not None:
                        df_anom.reset_index(drop=True, inplace=True)
                        x_anom = df_anom.iloc[:,:-4].to_numpy() 
                        x_arr = x_anom[:, ii]
                        y_arr = x_anom[:, jj]
                        for mm in range(len(x_arr)):
                            pgrid.axes[jj,ii].scatter(x_arr[mm], y_arr[mm], color=colors[ff], marker='$'+str(mm)+'$', s=20, zorder=3)
        for kk in range(ii):
            pgrid.axes[ii,kk].axhline(up_lim, color='green', linestyle='--')
            pgrid.axes[ii,kk].axhline(low_lim, color='red', linestyle='--')
            pgrid.axes[ii, kk].axhline(init_lim, color='red', linestyle=':')

    plt.tight_layout()
    plt.savefig(join(dirn, 'outliers_SR.pdf'), dpi=300)

    plt.close()
    pgrid = sns.pairplot(df_train.iloc[:, SRs:SRs+CRs], corner=True, kind='hist', plot_kws=dict(bins=nbins), diag_kws=dict(bins=nbins))
    p0s = np.array(metadata['starting_points'])
    for ii in range(len(pgrid.axes)):
        low_lim = metadata['lower_limits'][SRs+ii]
        up_lim = metadata['upper_limits'][SRs+ii]
        chan, obs = metadata['obs_yields'][SRs+ii]
        pgrid.axes[ii,ii].axvline(obs, color='orange', linestyle='--')
        pgrid.axes[ii,ii].xaxis.set_label_text(df_train.columns[SRs+ii], fontsize=10, visible=True)
        in_the_box_X = (df_train.iloc[:, SRs+ii] >= low_lim) & (df_train.iloc[:, SRs+ii] <= up_lim)
        for jj in range(ii, len(pgrid.axes)):
            pgrid.axes[jj,ii].axvline(up_lim, color='green', linestyle='--')
            pgrid.axes[jj,ii].axvline(low_lim, color='red', linestyle='--')
            pgrid.axes[jj,ii].set_ylabel(pgrid.axes[jj,ii].get_ylabel(), fontsize=10, visible=True)
            pgrid.axes[jj,ii].xaxis.set_label_text(pgrid.axes[jj,ii].get_xlabel(), fontsize=10, visible=True)
            in_the_box_Y = (df_train.iloc[:, SRs+jj] >=  metadata['lower_limits'][SRs+jj]) & (df_train.iloc[:, SRs+jj] <= metadata['upper_limits'][SRs+jj])
            in_the_box_N = np.sum(np.logical_and(in_the_box_X, in_the_box_Y))
            in_the_box_perc = np.round(100.0*in_the_box_N/df_train.iloc[:, SRs+jj].count(), 2)
            pgrid.axes[jj,ii].set_title('{:.2f}%'.format(in_the_box_perc), fontsize=8)

            if jj != ii:
                pgrid.axes[jj,ii].scatter(obs, metadata['obs_yields'][SRs+jj][1], color='orange', marker='*', zorder=3,)
                for ff in range(6):
                    # read the outliers info
                    df_anom = outlier_data[ff]
                    if df_anom is not None:
                        df_anom.reset_index(drop=True, inplace=True)
                        x_anom = df_anom.iloc[:,:-4].to_numpy() 
                        x_arr = x_anom[:, SRs+ii]
                        y_arr = x_anom[:, SRs+jj]
                        for mm in range(len(x_arr)):
                            pgrid.axes[jj,ii].scatter(x_arr[mm], y_arr[mm], color=colors[ff], marker='$'+str(mm)+'$', s=20, zorder=3)
        for kk in range(ii):
            pgrid.axes[ii,kk].axhline(up_lim, color='green', linestyle='--')
            pgrid.axes[ii,kk].axhline(low_lim, color='red', linestyle='--')
            #pgrid.axes[ii,kk].axhline(obs, color='orange')
    plt.tight_layout()
    plt.savefig(join(dirn, 'outliers_CR.pdf'), dpi=300)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('python plot_outliers.py <train_csv_path> <outliers_folder_path> <metadata_json_path>')
        exit(1)
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
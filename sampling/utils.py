#
# author: Rafal Maselek
# e-mail: rafal.maselek@lpsc.in2p3.fr
# 
# This file provides some functions used in sample.py
#

import numpy as np 
from prettytable import PrettyTable
from os.path import dirname, join, isfile, basename
import os
from misc import *
import copy
import pandas as pd
from name_dict import analysis_name_dict

def get2nd(arr):
    return np.array([b for a,b in arr])


def get_mask(shape, channels_and_bins, scan_SRs, scan_CRs, scan_VRs):
    mask = np.empty(shape, dtype=bool)
    ii = 0
    for c, t, b in channels_and_bins:
        fill_with = False
        if (t=='SR' and scan_SRs) or (t=='CR' and scan_CRs) or (t=='VR' and scan_VRs):
            fill_with = True
        mask[ii:ii+b] = fill_with
        ii += b
    return mask 


def print_yield_table(bins_names, bins_is_signal, bkg_yields, bkg_unc, obs_yields, logger):
    logger.info('Yields:')
    table = PrettyTable()
    table.add_column('BIN', bins_names, align='l', valign='t')
    table.add_column('SR', bins_is_signal, align='c', valign='t')
    table.add_column('BKG', [np.round(b,1) for a,b in bkg_yields], align='r', valign='t')
    table.add_column('∆BKG', [np.round(b,1) for a,b in bkg_unc], align='r', valign='t')
    table.add_column('OBS', [np.round(b,1) for a,b in obs_yields], align='r', valign='t')
    logger.info('\n'+table.get_string())
    return table.get_string()


def print_limit_table(bins_names, nSmin, nSmax, central_values, logger):
    logger.info('Scan limits:')
    table = PrettyTable()
    table.add_column('BIN', bins_names, align='l', valign='t')
    table.add_column('MIN', np.round(central_values+nSmin, 1), align='r', valign='t')
    table.add_column('MAX', np.round(central_values+nSmax, 1), align='r', valign='t')
    logger.info('\n'+table.get_string())
    return table.get_string()


def get_time_string(ns_t):
    s_t = ns_t//10**9
    m_t = s_t // 60
    h_t = m_t // 60
    left_s = s_t % 60
    left_m = m_t - 60*h_t
    return f"{h_t} hours {left_m} minutes {left_s} seconds"

def get_obs_signal(bkg_yields, obs_yields):
    obs_signal_yields = []
    for bkg_name, bkg_values in bkg_yields:
        for obs_name, obs_values in obs_yields:
            if obs_name == bkg_name:
                if type(obs_values) != type(list()):
                    obs_values = [obs_values]
                    bkg_values = [bkg_values]
                entry = (obs_name, [ a-b for a,b in zip(obs_values,bkg_values) ] )
                obs_signal_yields.append(entry)
                break

def get_scan_limits(bkg_yields, bkg_unc, obs_yields, channels_and_bins, signal_leakage_CR, signal_leakage_VR, CRs_scan_spread, VRs_scan_spread, CR_scan_sign, VR_scan_sign, CR_center_type, VR_center_type, logger):
    B, deltaB, obs = get2nd(bkg_yields), get2nd(bkg_unc), get2nd(obs_yields)

    # safety check
    for a,b,c in zip(bkg_yields, bkg_unc, obs_yields):
        if a[0]!=b[0] or a[0]!=c[0]:
            mes =  '[ERROR] order of channels is mixed between B, ∆B and Obs!'
            logger.critical(mes)
            raise ValueError(mes)

    # sometime the fitting procedure fails and returns too large blg unc 
    uncertainty_too_big = deltaB > 3 * np.sqrt(B)
    all_bins_names = [f"{c}-{b}" for c, _, binN in channels_and_bins for b in range(binN) ]
    if np.sum(uncertainty_too_big) > 0:
        bins_names = []
        ii = 0
        for c, t, b in channels_and_bins:
            for bb in range(b):
                if uncertainty_too_big[ii]:
                    bins_names.append(f"{c}-{b}")
                ii += 1
        logger.warning(f'Background uncertainty seems to large. I will clip it to 3√B for {bins_names}.')
        deltaB[uncertainty_too_big] = 3*np.sqrt(B[uncertainty_too_big])
    
    # calculate upper and lower limits
    nsMax = obs - (B-5*deltaB)
    nsMax = np.where(nsMax > 0.0, nsMax, 0.0)
    
    nsMin = -B+deltaB
    nsMin = np.where( nsMin > obs-B, obs-B, nsMin )
    nsMin = np.where(nsMin < 0.0, nsMin, -B)

    if CR_center_type == 'obs':
        nSobs_abs = np.abs(obs-B) # get the observed signal
    elif CR_center_type == 'exp':
        nSobs_abs = np.zeros(obs.shape) # use 0 signal
    else:
        raise ValueError(f"Central values for CRs should be either 'exp' or 'obs' but '{CR_center_type}' provided")

    CRmask = get_mask(len(B), channels_and_bins, False, True, False)
    if signal_leakage_CR:
        deltaS_CR = obs[CRmask]*CRs_scan_spread
        fluctuations = deltaS_CR - nSobs_abs[CRmask]
        if np.sum(fluctuations < 0.0, dtype=int) > 0:
            logger.warning('Fluctuations in the CRs are too small to account for the difference between expected and observed yields!')
            low_index = np.argmin(fluctuations)
            low_fluct_val = np.abs(nSobs_abs[CRmask][low_index]/obs[CRmask][low_index]) if obs[CRmask][low_index] > 0.0 else 0.0
            logger.warning(f'Consider increasing the "signal_leakage_CR_spread" to at least {low_fluct_val} (based on {all_bins_names[low_index]}).')
        # set the limits
        if CR_scan_sign == 'both':
            nsMax[CRmask] = deltaS_CR
            nsMin[CRmask] = -1.0*deltaS_CR  
        elif CR_scan_sign == 'positive':
            nsMax[CRmask] = deltaS_CR
            nsMin[CRmask] = -1e-10
        elif CR_scan_sign == 'negative':
            nsMax[CRmask] = 1e-10
            nsMin[CRmask] = -1.0*deltaS_CR
        else:
            mes =  f'[ERROR] Wrong value for the signal_leakage_CR_sign parameter: {CR_scan_sign}'
            logger.critical(mes)
            raise ValueError(mes)
    else:
        nsMax[CRmask] = 1e-10
        nsMin[CRmask] = -1e-10

    VRmask = get_mask(len(B), channels_and_bins, False, False, True)
    if signal_leakage_VR:
        # overwrite
        if VR_center_type == 'obs':
            nSobs_abs = np.abs(obs-B) # get the observed signal
        elif VR_center_type == 'exp':
            nSobs_abs = np.zeros(obs.shape) # use 0 signal
        else:
            raise ValueError(f"Central values for VRs should be either 'exp' or 'obs' but '{VR_center_type}' provided")

        deltaS_VR = obs[VRmask]*VRs_scan_spread
        fluctuations = deltaS_VR - nSobs_abs[VRmask]
        if np.sum(fluctuations < 0.0, dtype=int) > 0:
            logger.warning('Fluctuations in the VRs are too small to account for the difference between expected and observed yields!')
            low_index = np.argmin(fluctuations)
            low_fluct_val = np.abs(nSobs_abs[VRmask][low_index]/obs[VRmask][low_index]) if obs[VRmask][low_index] > 0.0 else 0.0
            logger.warning(f'Consider increasing the "signal_leakage_VR_spread" to at least {low_fluct_val} (based on {all_bins_names[low_index]}).')
        # set the limits
        if VR_scan_sign == 'both':
            nsMax[VRmask] = deltaS_VR
            nsMin[VRmask] = -deltaS_VR
        elif VR_scan_sign == 'positive':
            nsMax[VRmask] = deltaS_VR
            nsMin[VRmask] = -1e-10
        elif VR_scan_sign == 'negative':
            nsMax[VRmask] = 1e-10
            nsMin[VRmask] = -deltaS_VR
        else:
            mes =  f'[ERROR] Wrong value for the signal_leakage_VR_sign parameter: {VR_scan_sign}'
            logger.critical(mes)
            raise ValueError(mes)
    else:
        nsMax[VRmask] = 1e-10
        nsMin[VRmask] = -1e-10
        
    SRmask = get_mask(len(B), channels_and_bins, True, False, False)
    CRandVRmask = CRmask + VRmask
    central_values = np.empty(len(SRmask))
    for cc in range(len(central_values)):
        if SRmask[cc]:
            central_values[cc] = B[cc]
        elif CRandVRmask[cc]:
             central_values[cc] = obs[cc]
        else:
            mes =  f'[ERROR] wrong mask on position {cc}!'
            logger.critical(mes)
            raise ValueError(mes)

    assert central_values.shape[0] == nsMax.shape[0]
    return np.round(nsMin,4), np.round(nsMax,4), central_values


def find_mu_limits(nSmin, nSmax, central_values, logger):
    mu_min = np.max(central_values/(nSmin+1e-10))
    x = [ (nSmax[i]+central_values[i])/(nSmax[i]+1e-10) for i in range(len(central_values)) ]
    y = [ (x[i]-central_values[i]/(nSmax[i]+1e-10)) for i in range(len(central_values)) ]
    mu_max = np.min(y)
    logger.debug(f'Mu bounds initial estimate: ({mu_min}, {mu_max})')
    logger.debug(f'Central values: {central_values}')
    logger.debug(f'nSmin: {nSmin}')
    logger.debug(f'Central nSmax: {nSmax}')
    if mu_min >= mu_max:
        logger.warning(f'mu_min ({mu_min}) >= mu_max ({mu_max}) ! Setting initial limits to (0, 1).')
        mu_min = 0
        mu_max = 1
    return mu_min, mu_max


def generate_starting_points(nsMin, nsMax, central_values, mask, n=1, start_method='default', channels_and_bins=None, logger=None, starting_points_file=None, starting_points_file_index=None 
):
    #
    # some useful functions
    #
    def populate_randomly(n, mask, nsMin, nsMax):
        random_points = np.empty(shape=(n, len(nsMin)))
        for pp in range(n):
            random_points[pp] = np.array([np.random.uniform(nsMin[j],nsMax[j]) if mask[j] else 0.0 for j in range(len(nsMax))])
        return random_points

    def populate_gauss(n, mask, nsMin, nsMax):
        centers = np.zeros(len(mask))
        sigmas = 0.01 * (nsMax-nsMin)
        sigmas = np.where(sigmas > 1.0, 1.0, sigmas)
        cov = np.diag(sigmas**2)        
        random_points = np.random.multivariate_normal(centers, cov, size=n)
        random_points = np.clip(random_points, nsMin, np.inf)
        random_points[:, ~mask] = 0.0
        return random_points

    def populate_with_edges(n, mask, nsMin, nsMax):
        edge_points = np.empty(shape=(n, len(nsMin)))
        for pp in range(n):
            edge_points[pp] = np.array([np.random.choice([nsMin[j],nsMax[j]]) if mask[j] else 0.0 for j in range(len(nsMax))])
        return edge_points
    #
    # the actual generation
    #
    if logger is None:
        logger = setup_logger()
    
    if starting_points_file is not None:
        try:
            logger.warning('Using external file for starting points. Make sure it contains TOTAL yields.')
            logger.info(f'Attempting to read starting points from {starting_points_file}')
            table = pd.read_csv(starting_points_file)

            # 1 if pandas only saw one column and it contains ";" in its name, re-read assuming ";" is the separator
            if len(table.columns) == 1 and ";" in table.columns[0]:
                table = pd.read_csv(starting_points_file, sep=";")

            if starting_points_file_index is not None:
                idx = int(starting_points_file_index)
                sub = table.iloc[idx, :]
                # if you grabbed a single row (Series), turn it back into a 1×N DataFrame
                if isinstance(sub, pd.Series):
                    table = sub.to_frame().T
                else:
                    table = sub
            
            expected_cols = [ f"{channel}-{b}" for channel, sr, bins in channels_and_bins for b in range(bins)]

            # check for missing
            missing = set(expected_cols) - set(table.columns)
            if missing:
                logger.error(f"Missing columns in starting‐points file: {sorted(missing)}. I will use some default values for these.")
                for ii, colname in enumerate(expected_cols):
                    if colname in missing:
                        table.insert(ii, colname, central_values[ii], allow_duplicates=False)
                new_missing = set(expected_cols) - set(table.columns)
                if new_missing:
                    raise ValueError(f'Failure to add default values for columns: {new_missing}')

            # drop extras and reorder
            table = table[expected_cols]

            # extract raw numpy (no index, no names)
            points = table.to_numpy(dtype=float) - central_values
            logger.info('Starting points read!')
            return points

        except FileNotFoundError:
            raise FileNotFoundError('The file with starting points does not exist!')
        except Exception as e:
            raise e

    elif start_method == 'default':
        points = np.array([list(nsMin), list(nsMax), np.zeros(nsMin.shape)]) #nsMin is negative!
        if n > 3:
            random_points = populate_randomly(n-3, mask)
            points = np.concatenate([points, random_points], axis=0)
            assert len(points) == n
            return points
        else:
            return np.array(points[:n])
    elif start_method == 'random':
        points = populate_randomly(n, mask, nsMin, nsMax)
        assert len(points) == n
        return points 
    elif start_method == 'fine-tune':
        points = populate_gauss(n, mask, nsMin, nsMax)
        assert len(points) == n
        return points 
    elif start_method == 'edges':
        maxiter = np.min([n, 2**len(nsMin)])
        types_of_channels = set([ em[1] for em in channels_and_bins])
        if 'SR' not in types_of_channels:
            mes =  f'[ERROR] No signal regions provided! ({types_of_channels})'
            logger.critical(mes)
            raise ValueError(mes)
        else:
            edge_points = populate_with_edges(maxiter, mask, nsMin, nsMax)
            if n < maxiter:
                assert len(edge_points) == n 
                return edge_points
            else:
                generate_n = maxiter-n
                random_points = populate_randomly(generate_n, mask, nsMin, nsMax)
                points = np.concatenate([edge_points, random_points], axis=0)
                assert len(points) == n
                return points
    else:
        mes =  f"[ERROR] Wrong start method ({start_method}). I will use the 'default' option."
        logger.error(mes)
        return generate_starting_points(nsMin, nsMax, n, start_method='default', SRs=SRs, scan_CRs=scan_CRs)


def merge_results(infiles, keep_files=True, suffix='', logger=None):
    if logger is None:
        logger = setup_logger()
    logger.info(f'Merging results of {len(infiles)} scans.')
    try:
        ii = 0
        while True:
            dirpath = join(dirname(infiles[0]), "results-{}.csv".format(ii))
            if not isfile(dirpath):
                break
            else:
                ii += 1
    except PermissionError:
        # cannot create directory
        mes = '[ERROR] Cannot create output file with scan result. Permission denied.'
        logger.critical(mes)
        raise PermissionError(mes)
    else:
        if len(suffix) > 0:
            outpath = join(dirname(infiles[0]), "results-{}-{}.csv".format(suffix, ii))
        else:
            outpath = join(dirname(infiles[0]), "results-{}.csv".format(ii))
        min_values = []
        max_values = []
        with open(outpath, 'a') as fout:
            with open(infiles[0], 'r') as fin:
                fout.write(fin.readline())
            for ff, fpath in enumerate(infiles):
                with open(fpath, 'r') as fin:
                    loaded_data = np.loadtxt(fin, float, skiprows=1, delimiter=',')
                    if len(loaded_data.shape) == 1:
                        loaded_data = np.reshape(loaded_data, (1, loaded_data.shape[0]))
                    min_values.append(np.amin(loaded_data, axis=0))
                    max_values.append(np.amax(loaded_data, axis=0))
                    np.savetxt(fout, loaded_data, fmt="%+010.8f", delimiter=',')
                if not keep_files:
                    try:
                        os.remove(fpath)
                    except FileNotFoundError:
                        logger.error(f"File {basename(fpath)} cannot be deleted because it doesn't exist!.")
        min_values = np.array(min_values)
        max_values = np.array(max_values)
        return outpath, np.amin(min_values, axis=0).tolist(), np.amax(max_values, axis=0).tolist()


def create_metadata(param_dict, bkg_yields, bkg_unc, obs_yields, lower_limits, upper_limits, minS_orig, points, logger):
    metadata = copy.deepcopy(param_dict)
    if param_dict['analysis'] in analysis_name_dict.keys():
        metadata['analysis_altname'] = analysis_name_dict[param_dict['analysis']]
    elif param_dict['analysis'].split('-')[0] in analysis_name_dict.keys():
        metadata['analysis_altname'] = analysis_name_dict[param_dict['analysis'].split('-')[0]]
    else:
        logger.warning(f"Could not find alternative name for {param_dict['analysis']}")
        metadata['analysis_altname'] = ''
    metadata['bkg_yields'] = bkg_yields
    metadata['bkg_unc'] = bkg_unc
    metadata['obs_yields'] = obs_yields
    metadata['lower_limits'] = list(lower_limits)
    metadata['upper_limits'] = list(upper_limits)
    metadata['initial_lower_limits'] = list(minS_orig)
    metadata['starting_points'] = points
    return metadata


def update_metadata(metadata, data_min, data_max, nLL_max):  
    metadata['x_min'] = data_min[:-8]
    metadata['y_min'] = data_min[-8:]
    metadata['x_max'] = data_max[:-8]
    metadata['y_max'] = data_max[-8:]
    metadata['nLL_exp_max'] = []
    metadata['nLL_obs_max'] = []
    metadata['nLLA_exp_max'] = []
    metadata['nLLA_obs_max'] = []
    if nLL_max is not None:
        if len(nLL_max) != 4:
            raise ValueError('nLL_max should be a list of 4 values!')
        metadata['nLL_exp_max'] = [float(em) if em is not None else None for em in nLL_max[0]]
        metadata['nLL_obs_max'] = [float(em) if em is not None else None for em in nLL_max[1]]
        metadata['nLLA_exp_max'] = [float(em) if em is not None else None for em in nLL_max[2]]
        metadata['nLLA_obs_max'] = [float(em) if em is not None else None for em in nLL_max[3]]
    return metadata


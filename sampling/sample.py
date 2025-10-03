#
# author: Rafal Maselek
# e-mail: rafal.maselek@lpsc.in2p3.fr
# 
# This file is used to generate likelihood samples.
#
import os, sys
if 'NUMEXPR_MAX_THREADS' not in os.environ:
    os.environ['NUMEXPR_MAX_THREADS'] = '8' # silence numpy warning
import warnings
import yaml
from os.path import join, dirname, isfile, basename
import spey
import numpy as np
import time
import multiprocessing as mp
import json
from utils import *
from likelihood import *
from datetime import datetime, timedelta
import time
import copy
import random
from tqdm_loggable.auto import tqdm
from collections import OrderedDict
from misc import *
import tarfile
import gc
import jax
import argparse
from scipy.linalg import LinAlgError
from default_params import default_param_dict
jax.config.update('jax_platforms', 'cpu')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# YAML parsing
#
def include_handler(data):
    """
    Handle the `include` key to load and unpack the referenced YAML file.
    """
    if not isinstance(data, dict):
        return data

    # Check if the "include" key exists
    if "include" in data:
        file_path = data["include"]
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Included file {file_path} not found.")
        with open(file_path, 'r') as f:
            included_data = yaml.load(f, Loader=yaml.FullLoader)  # Using FullLoader directly
            if not isinstance(included_data, dict):
                raise ValueError(f"Included file {file_path} must contain a dictionary.")
            del data["include"]  # Remove the `include` key
            data.update(included_data)  # Merge included data into the main dictionary
    return data

def load_yaml_with_includes(file_path):
    """
    Load YAML documents, processing the `include` key for each document.
    """
    with open(file_path, 'r') as f:
        documents = yaml.full_load_all(f)  # Using FullLoader directly
        return [include_handler(doc) for doc in documents]

#
# MAIN
#
def main(logger, param_file, starting_points_file, starting_points_file_index):
    logger = logging.getLogger("main_logger")

    yaml_docs = None
    param_docs = []
    # default values of parameters
    global_param_dict = default_param_dict

    if os.path.isfile(param_file):
        # with open(param_file, 'r') as f:
        #     config = yaml.full_load_all(f,  Loader=IncludeLoader)
        #     yaml_docs = [em for em in config]
        yaml_docs = load_yaml_with_includes(param_file)

        # consitency checks
        if len(yaml_docs) == 0:
            mes = 'No yaml documents found inside the param file!'
            logger.critical(mes)
            raise ValueError(mes)
        elif len(yaml_docs) == 1:
            if yaml_docs[0].get('analyses') is not None:
                mes = 'Parameter should contain the list of analyses to read and subsequently their parameters in seperate yaml documents!'
                logger.critical(mes)
                raise ValueError(mes)
            else:
                # there is only one analysis, parse it
                logger.warning('Parameter file seems to contain a single analysis. Loading..')
                name = yaml_docs[0].get('name')
                global_param_dict['analyses'] = [name]
                param_docs = yaml_docs
        else:
            if yaml_docs[0].get('analyses') is None:
                logger.warning('List of analyses not provided! I will scan all available.')
            else:
                for k,v in yaml_docs[0].items():
                    global_param_dict[k] = v
                param_docs = yaml_docs[1:]
    else:
        mes = 'Parameter file "parameters.yaml" is missing!'
        logger.critical(mes)
        raise FileNotFoundError(mes)
    if yaml_docs is None:
        mes = 'Unable to parse parameter file. Aborting.'
        logger.critical(mes)
        raise RuntimeError(mes)

    del yaml_docs
    spey.set_log_level(global_param_dict['spey_verbose_lvl'])
    logger.info(f'Param file loaded. {len(global_param_dict["analyses"])} analyses to be sampled.')    
    if len(list(set(global_param_dict['analyses']))) > len(param_docs):
        logger.warning(f'The list of analyses to sample is longer than the list of parameter documents!')
    
    #############################################
    # create a directory for likelihood tables
    #############################################
    tables_dir = "../tables"
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir, exist_ok=True)
        logger.info('Created parent directory for likelihood tables.')
    else:
        logger.info('Parent directory for likelihood tables already exists.')

    #########################
    # loop over the analyses
    #########################
    global_param_dict['merged'] = False # add metadata with info about the post-sampling merging of results
    for ii in range(len(param_docs)):
        analysis = param_docs[ii]
        param_dict = copy.deepcopy(global_param_dict)
        for k,v in analysis.items():
            param_dict[k] = v
        name = param_dict['analysis']
        if global_param_dict['analyses']=='all' or name in global_param_dict['analyses']:
            ###############################
            # sample the selected analysis
            ###############################
            analysis_time = time.process_time_ns()
            logger.info(f'Working on analysis {name}.')
            # check for the input folder and unpack the archive if needed
            input_folder = join(param_dict['input_folder'], name)
            if not os.path.isdir(input_folder):
                archive_path = join(param_dict['input_folder'], name+'.tar.gz')
                if os.path.exists(archive_path):
                    logger.info(f'Unpacking {name}.tar.gz')
                    try:
                        with tarfile.open(archive_path, 'r:gz') as tar:
                            tar.extractall(path=param_dict['input_folder'])
                    except Exception as e:
                        logger.error("Couldn't unpack archive with likelihood model! Message: " +str(e))
                        continue
                else:
                    logger.error(f'Input files not found for {name}.')
                    continue

            # set the random seed
            if 'seed' not in param_dict.keys():
                param_dict['seed'] = int(time.time()) % (2**32-1)

            # set up slurm computations 
            if param_dict['cluster']:
                logger.info('Running in the cluster mode!')
                slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
                slurm_task_id = int(slurm_task_id) if slurm_task_id and slurm_task_id.isdigit() else 0
                param_dict['seed'] = slurm_task_id + param_dict['seed']

            logger.info(f'Using random seed value: {param_dict["seed"]}')
            set_global_determinism(seed=param_dict['seed'])

            try:
                ii = 0
                while True:
                    if not param_dict['cluster']: 
                        dirpath = join(param_dict['output_folder'], "{}-{}".format(name, ii) )
                    else:
                        dirpath = join(param_dict['output_folder'], "{}-{}-{}".format(name, ii, slurm_task_id) )  
                    
                    if not os.path.isdir(dirpath):
                        break
                    else:
                        ii += 1
                os.makedirs(dirpath)
                logger.info('Created directory for likelihood tables.')
            except PermissionError:
                # cannot create directory
                mes = 'Cannot create output directory. Permission denied.'
                logger.critical(mes)
                raise PermissionError(mes)

            bkgfiles = param_dict['bkgfiles']
            if bkgfiles is None:
                mes = 'Background files missing in the parameter file!'
                logger.critical(mes)
                raise ValueError(mes)                
            
            ####################
            # Load the patches
            ####################
            patchsets = param_dict['patchsets']
            if patchsets is None:
                mes = 'Provide at least one patchset in the parameter file!'
                logger.critical(mes)
                raise ValueError(mes)
            
            ######################################################################
            # Special case when one background file corresponds to many patchsets
            ######################################################################
            if len(patchsets) > 1 and len(bkgfiles) == 1:
                # we simply emulate provision of multiple background files
                bkgfiles = [ bkgfiles[0] ] * len(patchsets)
                # works for 1-1 and 1-many
                param_dict['bkg_yields'] = [param_dict['bkg_yields']] * len(patchsets)
                param_dict['bkg_unc'] = [param_dict['bkg_unc']] * len(patchsets)
            elif len(patchsets) == 1 and len(bkgfiles) == 1:
                 param_dict['bkg_yields'] = [ param_dict['bkg_yields'] ]
                 param_dict['bkg_unc'] = [ param_dict['bkg_unc'] ]
            # Check if each patchset is matched with background file
            if len(patchsets) != len(bkgfiles):
                    mes = f'Number of patchsets ({len(patchsets)}) does not match the number of background files ({len(bkgfiles)})!'
                    logger.critical(mes)
                    raise ValueError(mes)
            
            for file_pair_index, (bkgfile, patchset_var) in enumerate(zip(bkgfiles, patchsets)):
                if isinstance(patchset_var, str):
                    patchset_path = patchset_var
                elif isinstance(patchset_var, list):
                    patchset_path = patchset_var[0]
                    if not patchset_var[1]:
                        mes = f'Skipping {patchset_path}'
                        logger.warning(mes)
                        continue
                else:
                    mes = f'Patchset entries should be strings or lists, but {type(patchset_var)} provided!'
                    logger.critical(mes)
                    raise TypeError(mes)


                logger.info(f"Analysing {basename(bkgfile)} and {basename(patchset_path)} files [{file_pair_index+1}/{len(bkgfiles)}].")
                suffix = basename(bkgfile).split('.')[0]+'-'+basename(patchset_path).split('.')[0]

                # load the channels for given pair of files
                if len(param_dict['bkgfiles']) == 1:
                    if len(param_dict['channels']) > 1:
                        logger.warning(f'There are more groups of channels ({len(param_dict["channels"])}) specified than input background files. I will ignore additional channel groups.')
                        channels = OrderedDict(param_dict['channels'][file_pair_index])
                    else:
                        channels = OrderedDict(param_dict['channels'][0])
                else:
                    channels = OrderedDict(param_dict['channels'][file_pair_index])
                SRs = [k for k,v in channels.items() if v=='SR']


                with open(join(param_dict['input_folder'], name, patchset_path), 'r') as fpatch:
                    patch_json = json.load(fpatch)
                patchset_spec = pyhf.PatchSet(patch_json)
                logger.info(f'{len(patchset_spec.patches)} patches in the analysis.')
                logger.info(f'{len(channels.keys())} channels found with {len(SRs)} signal regions: \n\t\t\t'+'\n\t\t\t'.join(SRs))
               
                ##############################
                # determine background values
                ##############################
                logger.info(f"Getting the background yields.")
                with open(join(param_dict['input_folder'], name, bkgfile)) as serialized:
                    bkg_spec = json.load(serialized)
                # bkgonly = pyhf.Workspace(bkg_spec)
                stat_wrapper = spey.get_backend("pyhf")
                signal_patch = patch_json["patches"][0]["patch"]
                interpreter = WorkspaceInterpreter(bkg_spec)
                signal_map, modifiers = interpreter.patch_to_map(signal_patch=signal_patch)
                for key, item in signal_map.items():
                    interpreter.inject_signal(key, item)
                stat_wrapper = spey.get_backend("pyhf")
                full_background_model = stat_wrapper(
                                                    analysis=name,
                                                    background_only_model=bkg_spec,
                                                    signal_patch=interpreter.make_patch(),
                                                )
                _, model_bkg, data_bkg = full_background_model.backend.model(expected=spey.ExpectationType.apriori)
                
                # extract channels
                if set(model_bkg.config.channel_nbins.keys()) != set(channels.keys()):
                    mes = 'Channels found in the background file does not match channels provided in the parameter file!'
                    logger.critical(mes)
                    raise ValueError(mes)

                bins_names = []
                bins_is_signal = []
                channels_and_bins = []
                for k,v in model_bkg.config.channel_nbins.items():
                    channels_and_bins.append((k, channels[k], v))
                    for ii in range(v):
                        bins_names.append(k+'-{}'.format(ii))
                        bins_is_signal.append(k in SRs)
                bins_is_signal = np.array(bins_is_signal)
                bkg_yields = list(zip(bins_names, data_bkg[:len(bins_is_signal)]))
                del data_bkg
                del model_bkg
                if not param_dict['fit_bkg']:
                    logger.info(f"Loading background yields and uncertainties from the file.")
                    file_data_bkg = param_dict['bkg_yields'][file_pair_index]
                    bkg_yields = list(zip(bins_names, file_data_bkg))
                    file_data_bkg_unc = param_dict['bkg_unc'][file_pair_index]
                    bkg_unc = list(zip(bins_names, file_data_bkg_unc))
                    if len(bins_names) != len(file_data_bkg) or  len(bins_names) != len(file_data_bkg_unc):
                        mes = f'There is a mismatch between the number of recognised bins ({len(bins_names)}), \
                        the number of bkg yields ({len(file_data_bkg)}), and the number of bkg uncertainties ({len(file_data_bkg_unc)}). Please proceed carefully.'
                        logger.warning(mes)
                else:
                    ########################################
                    # determine background uncertainties
                    ########################################
                    full_background_model.backend.manager.backend = "jax"
                    # need to convert full background model to simplified in order to get systematic uncertainty on B
                    converter = spey.get_backend("pyhf.simplify")
                    logger.info(f"Estimating background uncertainty ...")

                    n_bkg_unc = param_dict['bkg_unc_samples']
                    # control_region_indices=[val for val in list(interpreter.channels) if not val in SRs]
                    # this might fail so we make a loop
                    for trial in range(3):
                        try:
                            simplified_background_model = converter(
                                                    statistical_model=full_background_model,
                                                    convert_to="default_pdf.correlated_background",
                                                    control_region_indices=list(interpreter.channels), #control_region_indices,
                                                    number_of_samples=n_bkg_unc,
                                                    
                                                )
                            break
                        except LinAlgError as e:
                            mes = f'Conversion to simplified model failed: ' + repr(e)
                            logger.error(mes)
                            if trial == 2:
                                raise
                            else:
                                mes = f'Trying again...'
                                logger.warning(mes)

                    cov_matrix = simplified_background_model.backend.covariance_matrix
                    del simplified_background_model 
                    bkg_unc = list( zip( bins_names, list(np.sqrt(np.diag(cov_matrix))) ) )
                    del converter
                ##############################
                # determine the observed yields
                ##############################
                logger.info(f"Geting the observed number of events.")
                workspace_obs, model_obs, data_obs = full_background_model.backend.model(expected=spey.ExpectationType.observed)
                obs_yields = list(zip(bins_names, data_obs))
                print_yield_table(bins_names, bins_is_signal, bkg_yields, bkg_unc, obs_yields, logger)
                
                
                #################################################################
                # Check the validity of scan, points, and processes parameters
                #################################################################
                if param_dict['scans'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['scans']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)
                if param_dict['points'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['points']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)
                if param_dict['processes'] < 1:
                    mes = f"Number of scans has to be at least 1, but {param_dict['processes']} is provided!"
                    logger.critical(mes)
                    raise ValueError(mes)

                ##############################
                # determine limits of the scan
                ##############################
                logger.info(f"Setting initial scan limits on S.")

                mask = get_mask(len(obs_yields), channels_and_bins, True, param_dict['signal_leakage_CR'], param_dict['signal_leakage_VR'])
                nSmin, nSmax, central_values = get_scan_limits(bkg_yields, 
                                                                bkg_unc, 
                                                                obs_yields, 
                                                                channels_and_bins,
                                                                param_dict['signal_leakage_CR'], 
                                                                param_dict['signal_leakage_VR'],
                                                                param_dict['signal_leakage_CR_spread'],
                                                                param_dict['signal_leakage_VR_spread'],
                                                                param_dict['signal_leakage_CR_sign'],
                                                                param_dict['signal_leakage_VR_sign'],
                                                                param_dict['CR_center'],
                                                                param_dict['VR_center'],
                                                                logger
                                                                )
                nSobs = get_obs_signal(bkg_yields, obs_yields)
                nSmin_orig = copy.deepcopy(nSmin)
                logger.info(f"Setting the absolute lower limit on S ...")
                nSmin = find_min_S(param_dict['low_lim_samples'], bkg_spec, stat_wrapper, nSmin, channels_and_bins, logger)
                print_limit_table(bins_names, nSmin, nSmax, central_values, logger)
                # find mu_SIG limits for max likelihood calculation
                mu_min, mu_max = find_mu_limits(nSmin, nSmax, central_values, logger)
                poi_index = full_background_model.backend.config().poi_index
                mu_min_orig, mu_max_orig = full_background_model.backend.config().suggested_bounds[poi_index]
                mu_min = mu_min_orig if mu_min_orig < mu_min else mu_min
                mu_max = mu_max_orig if mu_max_orig > mu_max else mu_max
                mu_bounds = (mu_min, mu_max)
                logger.info(f'Mu bounds estimated: ({mu_min}, {mu_max})')
                del full_background_model

                logger.info(f"Generating initial points.")

                p0s = generate_starting_points(nSmin, nSmax, central_values,
                                    mask=mask,
                                    n=param_dict['scans'], 
                                    start_method=param_dict['start_method'], 
                                    channels_and_bins=channels_and_bins,
                                    starting_points_file=starting_points_file,
                                    starting_points_file_index=starting_points_file_index 
                                    )
                # When using input file with total yields, one has to subtract central values
                # if starting_points_file is not None:
                #     if len(p0s[0]) != len(central_values):
                #         raise IndexError(f'Size mismatch! Starting points: {len(p0s[0])}, expected: {len(central_values)}!')
                #     if len(p0s) > 1:
                #         raise NotImplementedError(f'Expected a single starting point from file, but encountered {len(p0s)}.')
                #     p0s[0] = p0s[0] - central_values

                ##############################
                # Prepare channels for removal
                ##############################
                if param_dict['removeCRsVRs']:
                    if (param_dict['signal_leakage_CR'] or param_dict['signal_leakage_VR']):
                        mes = f"Asked to remove CRs/VRs but signal leakage enabled! I don't know what to do so I abort!"
                        logger.critical(mes)
                        raise ValueError(mes)
                    elif param_dict['remove_channels'] is None:
                        channels_to_be_removed = []
                        for c, sr, b in channels_and_bins:
                            if sr != 'SR':
                                channels_to_be_removed.append(c)
                        param_dict['remove_channels'] = channels_to_be_removed
                        del channels_to_be_removed
                    else:
                        for c, sr, b in channels_and_bins:
                            if sr != 'SR':
                                param_dict['remove_channels'] =  param_dict['remove_channels'] + [c]
                # check if the channels to be removed are available
                param_dict['remove_channels'] = [c for c in param_dict['remove_channels'] if c in channels]                
                for channel_name in param_dict['remove_channels']:
                    logger.warning(f'Removing channel: {channel_name}')


                ####################
                ######  SCAN  ######
                ####################
                
                logger.info('Saving metadata.')
                metadata = create_metadata(param_dict,
                                            bkg_yields, 
                                            bkg_unc, 
                                            obs_yields, 
                                            central_values+nSmin, 
                                            central_values+nSmax, 
                                            central_values+nSmin_orig,
                                            [list(p + central_values) for p in p0s],
                                            logger)
                # save metadata in case of scan failure
                metadata_path = join(dirpath, 'metadata.json')
                with open(metadata_path, "w") as outfile: 
                    json.dump(metadata, outfile)
                del nSmin_orig
                logger.info('Preparing the scan.')
                sigmas = calculate_sigmas(nSmin, nSmax, mask, param_dict['SR_sigma'], param_dict['CR_sigma'], param_dict['VR_sigma'], channels_and_bins)

                scan_wrapper = ScanWrapper(param_dict['points'],
                                        bkg_spec, 
                                        sigmas,
                                        channels_and_bins, 
                                        central_values, 
                                        param_dict['buffer_size'], 
                                        nSmin, 
                                        param_dict['scan_criterion'],
                                        mu_bounds,
                                        seed=param_dict['seed'],
                                        remove_channels=param_dict['remove_channels'],
                                        logger=logger
                                        )
                #inputs = [(p, join(dirpath, f'table-{int(time.process_time_ns() - analysis_time ) + np.random.randint(1, 999)}.csv')) for p in p0s]
                inputs = [(p, join(dirpath, f'table-{suffix}-{ii}.csv')) for ii, p in enumerate(p0s)]
                del p0s
                gc.collect()
                logger.info(f"Running {len(inputs)} scans, {param_dict['points']} points each, with {param_dict['processes']} processes ...")

                nLL_max_results = []
                # pool
                with mp.Pool(processes=param_dict['processes'], maxtasksperchild=1) as pool:
                    for nLL_max in tqdm(pool.imap_unordered(scan_wrapper, inputs), total=len(inputs), colour='GREEN'):
                        nLL_max_results.append(nLL_max)

                # merge the results
                logger.info('Merging the results.')
                merged_file_path, min_values, max_values = merge_results([em[1] for em in inputs], keep_files = param_dict['keep_files'], suffix=suffix, logger=logger)
                logger.info(f'Results saved to {merged_file_path}')
                # update the metadata
                logger.info('Updating the metadata.')
                metadata = update_metadata(metadata, min_values, max_values, nLL_max_results[0])
                with open(metadata_path, "w") as outfile: 
                    json.dump(metadata, outfile, indent = 4)
                os.rename(metadata_path, merged_file_path.replace('.csv', '.json'))
                # calculate time of the scan
                final_time = time.process_time_ns() - analysis_time
                time_per_point = np.round( (final_time//10**9) / ( param_dict['points'] * param_dict['scans']), 3)
                time_string = get_time_string(final_time)
                logger.info(f'Finished scan for {param_dict["analysis"]}. It took {time_string} in total, on average {time_per_point} seconds per point.')


if __name__ == "__main__":
    """Main function for script execution."""
    parser = argparse.ArgumentParser(description="Run the script with a parameter file and optional log directory.")
    parser.add_argument("param_file", nargs="?", default="parameters.yaml", help="Path to the YAML file with parameters.")
    parser.add_argument("--log_dir", default="logs", help="Directory where logs should be saved.")
    parser.add_argument("--start", default=None, help="CSV file with starting points.")
    parser.add_argument("--start_index", default=None, help="Row to choose from the input file")

    args = parser.parse_args()

    # Initialize logger
    logger = setup_logger(args.log_dir)
    logger.info(f"Using log directory: {args.log_dir}")

    mp.set_start_method('spawn', force=True)
    logger.info("Process spawned.")

    logger.info(f"Loading the parameter file: {args.param_file}")


    start_time = datetime.now()
    if 'pyhf' not in spey.AvailableBackends():
        mes = 'PyHF backend was not found! Aborting.'
        logger.critical(mes)
        raise RuntimeError(mes)
    try:
        main(logger, args.param_file, args.start, args.start_index)
    except Exception as e:
        mes = f'Program failed: ' + repr(e)
        logger.critical(mes)
        raise Exception(mes)
    end_time = datetime.now()

    logger.info("[END] Program finished! Total time of execution {}".format(end_time - start_time))


#
# author: Rafal Maselek
# e-mail: rafal.maselek@lpsc.in2p3.fr
# 
# This file implements functions used by sample.py to perform likelihood sampling
#

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ.pop('TF_CONFIG', None)
import numpy as np
from math import isinf, isnan
from timeit import default_timer as timer
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow_probability as tfp
from misc import *
import pyhf
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
import os
from tqdm_loggable.auto import tqdm as progressbar
import warnings
import random
import gc



def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    

def set_global_determinism(seed):
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    set_seeds(seed=seed)



def find_min_S(niter, bkg_spec, stat_wrapper, nSmin, channels_and_bins, logger):
    if niter < 1:
        logger.warning(f"The value of the 'low_lim_samples' parameter is {niter} < 1! No negative signal in SRs will be injected.")
    
    #pyhf.set_backend('tensorflow')
    up_lim = np.zeros(nSmin.shape, dtype=float)
    minimalS = np.zeros(nSmin.shape, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nbins = np.sum([em[2] for em in channels_and_bins])
        ii = 0
        
        with progressbar(total=nbins*niter) as pbar:
            interpreter = WorkspaceInterpreter(bkg_spec)
            # iterate over all channels and bins
            for jj in range(len(channels_and_bins)):
                c, sr, b = channels_and_bins[jj]
                # extract signal value for given channel
                bin_vals = np.array(nSmin[ii:ii+b])
                # get the default signal for injection
                inject_vals = np.zeros(bin_vals.shape)
                # iterate over all bins in that channel
                for bb in range(b):
                    # optimise value of signal in the channel, start with mu=1   
                    mu = 1.0
                    mu_old = 1.0
                    mu_new = None
                    # perform niter steps to find the lower limit on signal
                    for nn in range(niter):
                        # set signal for given bin
                        inject_vals[bb] = np.round(bin_vals[bb] * mu+1e-4, 4) 
                        # inject the signal to all bins
                        interpreter.inject_signal(c, inject_vals)
                        statistical_model = stat_wrapper(
                                                    background_only_model=interpreter.background_only_model,
                                                    signal_patch=interpreter.make_patch(),
                                                    )
                        statistical_model.backend.manager.backend = "tensorflow"
                        nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')
                        #nLL_exp_mu1 = tf.squeeze(nLL_exp_mu1).numpy()
                        nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
                        #nLL_obs_mu1 = tf.squeeze(nLL_obs_mu1).numpy()
                        
                        # print(c, mu, bin_vals[bb] * mu, nLL_exp_mu1, nLL_obs_mu1, inject_vals)
                        if isnan(nLL_exp_mu1) or isnan(nLL_obs_mu1) or isinf(nLL_exp_mu1) or isinf(nLL_obs_mu1):
                            mu_old = mu
                            mu = mu/2.0
                        else:
                            minimalS[ii+bb] = inject_vals[bb] 
                            mu_new = (mu+mu_old)/2.0
                            if np.isclose(mu_new, 1.0, atol=1e-3):
                                break
                            mu = mu_new
                    pbar.update(niter)

                    if mu_new is None:
                        logger.warning(f'Not able to find lower limit on negative signal in {c}-{bb}. No negative signal in this bin will be used. Consider increasing "low_lim_samples" parameter.')
                        minimalS[ii+bb] = up_lim[ii+bb]
                ii += b
    del interpreter, inject_vals
    return minimalS


class NewStateWrapper():
    def __init__(self, std, minimalS_allowed):
        #self.nSmin = tf.convert_to_tensor(nSmin, dtype=float)
        #self.nSmax = tf.convert_to_tensor(nSmax, dtype=float)
        self.minimalS_allowed = minimalS_allowed
        self.std = std
        self.dims = np.shape(std)[0]
    def __call__(self, prev_state, seed):
        #a = tf.convert_to_tensor(a, dtype=float)
        new_state_dist = tfp.distributions.MultivariateNormalDiag(loc=prev_state, scale_diag=self.std*np.ones(self.dims))
        new_state = new_state_dist.sample(seed=seed)
        new_state = np.where(new_state>=self.minimalS_allowed, new_state, self.minimalS_allowed)
        del new_state_dist
        return new_state


class LikelihoodCalculatorWrapper():
    # create a callable object that will keep information about the background and construct patches from the S yields
    def __init__(self, bkg_spec, channels_and_bins, central_values, output_file, buff_size, criterion, mu_bounds, seed, remove_channels, logger):
        self._bkg_spec = bkg_spec
        self._channels_and_bins = channels_and_bins
        self._counter = 0
        self._output_file = output_file
        self._buff_size = buff_size
        self._central_values = central_values
        self._criterion = criterion
        self._mu_bounds = mu_bounds
        self._remove_channels = [] if remove_channels is None else remove_channels
        self.logger = logger
        # write the first line with header names
        bin_no = 0
        try:
            line = ''
            for c, sr, b in channels_and_bins:
                    for val in range(b):
                        if c not in self._remove_channels:
                            line = line + f'{c}-{val},'
                        bin_no += 1
            line = line + 'nLL_exp_mu0,nLL_exp_mu1,nLL_obs_mu0,nLL_obs_mu1,nLLA_exp_mu0,nLLA_exp_mu1,nLLA_obs_mu0,nLLA_obs_mu1\n'
            with open(self._output_file, 'w') as fout:
                fout.write(line)
        except FileExistsError:
            mes = '[ERROR] The output file {} already exists!'
            self.logger.critical(mes)
            raise FileExistsError(mes)

        self._bin_no = bin_no
        # handle the removed channels
        self._mask = np.ones(shape=self._bin_no, dtype=bool)
        cc = 0
        for c, sr, b in channels_and_bins:
            if c in self._remove_channels:
                self._bin_no = self._bin_no - b
                self._mask[cc] = False
            cc += 1

        self._results = np.empty(shape=(self._buff_size, self._bin_no+8), dtype=float)
        self.nLL_exp_mu0 = None
        self.nLL_obs_mu0 = None
        self.nLLA_exp_mu0 = None
        self.nLLA_obs_mu0 = None
        self._stat_wrapper = spey.get_backend("pyhf")
        self._seed = seed

        gc.enable()
        gc.set_threshold(350, 5, 5)


    def calculate_Lmu0(self, S_yields):
        self.logger.debug('Calculating nLL for mu=0.')
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)
        
        my_patch = interpreter.make_patch()
        self.logger.debug('First patch: {}'.format(my_patch))

        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=my_patch,
                                        )
        statistical_model.backend.manager.backend = "tensorflow"
        nLL_exp_mu0 = statistical_model.likelihood(poi_test=0.0, expected='apriori')
        nLL_exp_mu0 = self.check_for_nan(nLL_exp_mu0, 'nLL_exp_mu0')
        self.nLL_exp_mu0 = nLL_exp_mu0
        nLL_obs_mu0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
        nLL_obs_mu0 = self.check_for_nan(nLL_obs_mu0, 'nLL_obs_mu0')
        self.nLL_obs_mu0 = nLL_obs_mu0
        # Asimov likelihood
        nLLA_exp_mu0 = statistical_model.asimov_likelihood(poi_test=0.0, expected='apriori')
        nLLA_exp_mu0 = self.check_for_nan(nLLA_exp_mu0, 'nLLA_exp_mu0')
        self.nLLA_exp_mu0 = nLLA_exp_mu0
        nLLA_obs_mu0 = statistical_model.asimov_likelihood(poi_test=0.0, expected='observed')
        nLLA_obs_mu0 = self.check_for_nan(nLLA_obs_mu0, 'nLLA_obs_mu0')
        self.nLLA_obs_mu0 = nLLA_obs_mu0

        del interpreter, statistical_model
        gc.collect()

    def calculate_Lmax(self, S_yields):
        self.logger.info('Calculating maximum likelihood.')
        stat_wrapper = spey.get_backend("pyhf")
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)
        
        my_patch = interpreter.make_patch()
        self.logger.debug('First patch: {}'.format(my_patch))

        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=my_patch,
                                        )
        par_bounds = statistical_model.backend.config().suggested_bounds
        poi_index = statistical_model.backend.config().poi_index
        par_bounds[poi_index] = self._mu_bounds

        try:
            nLL_exp_max = statistical_model.maximize_likelihood(expected=spey.ExpectationType.apriori, par_bounds=par_bounds, )
        except Exception as e:
            self.logger.error('Failed to calculate nLL_exp_max: '+str(e))
            nLL_exp_max = [None, None]
        
        try:    
            nLL_obs_max = statistical_model.maximize_likelihood(par_bounds=par_bounds, )
        except Exception as e:
            self.logger.error('Failed to calculate nLL_obs_max: '+str(e))
            nLL_obs_max = [None, None]
        
        try:                            
            nLLA_exp_max = statistical_model.maximize_asimov_likelihood(test_statistics="qmutilde", expected=spey.ExpectationType.apriori, par_bounds=par_bounds,)
        except Exception as e:
            self.logger.error('Failed to calculate nLLA_exp_max: '+str(e))
            nLLA_exp_max = [None, None]
        
        try:
            nLLA_obs_max = statistical_model.maximize_asimov_likelihood(test_statistics="qmutilde", par_bounds=par_bounds)
        except Exception as e:
            self.logger.error('Failed to calculate nLLA_obs_max: '+str(e))
            nLLA_obs_max = [None, None]

        self.nLL_max = [list(nLL_exp_max), list(nLL_obs_max), list(nLLA_exp_max), list(nLLA_obs_max)]

    def clear_buffer(self):
        del self._results
        self._results = np.empty(shape=(self._buff_size, self._bin_no+8), dtype=float)
        self._counter = 0

    def inject_signal(self, interpreter, S_yields):
        ii = 0
        for c, sr, b in self._channels_and_bins:
            bin_vals = np.array(S_yields[ii:ii + b]) 
            interpreter.inject_signal(c, bin_vals)
            ii += b
        return interpreter  

    def save_results(self, counter=None):
        if counter is None:
            counter = self._counter
        with open(self._output_file, 'a') as fout:
                np.savetxt(fout, self._results[:counter], fmt="%+010.8f", delimiter=',')
        self.clear_buffer()
    
    def check_for_nan(self, likelihood, name):
        if isnan(likelihood):
            self.logger.error(f'[ERROR] {name} is {likelihood}! I will write it as +1e10')
            return np.float64(1e10)
        else:
            return likelihood
    
    def __call__(self, S_yields):
        set_seeds(self._seed)
        if self.nLL_exp_mu0 is None:
            # first call
            self.calculate_Lmu0(S_yields)
            self.calculate_Lmax(S_yields)
        interpreter = WorkspaceInterpreter(self._bkg_spec)
        interpreter = self.inject_signal(interpreter, S_yields)
        for channel_name in self._remove_channels:
            interpreter.remove_channel(channel_name)

        statistical_model = self._stat_wrapper(
                                            background_only_model=interpreter.background_only_model,
                                            signal_patch=interpreter.make_patch(),
                                        )
        statistical_model.backend.manager.backend = "tensorflow"
        nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')       
        nLL_exp_mu1 = self.check_for_nan(nLL_exp_mu1, 'nLL_exp_mu1')
        nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
        nLL_obs_mu1 = self.check_for_nan(nLL_obs_mu1, 'nLL_obs_mu1')
        # Asimov likelihoods
        nLLA_exp_mu1 = statistical_model.asimov_likelihood(poi_test=1.0, expected='apriori')       
        nLLA_exp_mu1 = self.check_for_nan(nLLA_exp_mu1, 'nLLA_exp_mu1')
        nLLA_obs_mu1 = statistical_model.asimov_likelihood(poi_test=1.0, expected='observed')
        nLLA_obs_mu1 = self.check_for_nan(nLLA_obs_mu1, 'nLLA_obs_mu1')

        likelihoods_to_save = [self.nLL_exp_mu0, nLL_exp_mu1, self.nLL_obs_mu0, nLL_obs_mu1, \
                                self.nLLA_exp_mu0, nLLA_exp_mu1, self.nLLA_obs_mu0, nLLA_obs_mu1]
        yields_to_save = list(np.array((self._central_values+S_yields))[self._mask])
        self._results[self._counter] = np.array( yields_to_save + likelihoods_to_save, dtype=float)
        self._counter += 1
        del interpreter, statistical_model
        if self._counter == self._buff_size:
            self.save_results()
            gc.collect()
        if self._criterion == 'nLL_obs_mu1':
            return -1*nLL_obs_mu1
        elif self._criterion == 'nLL_exp_mu1':
            return -1*nLL_exp_mu1
        elif self._criterion == 'LL_obs_mu1':
            return nLL_obs_mu1
        elif self._criterion == 'LL_exp_mu1':
            return nLL_exp_mu1
        else:
            mes = '[ERROR] Wrong criterion passed to LikelihoodCalculatorWrapper.'
            self.logger.critical(mes)
            raise ValueError(mes)
        del nLL_exp_mu1, nLL_obs_mu1, nLLA_exp_mu1, nLLA_obs_mu1,

    def get_counter(self):
        return self._counter

def scan(p0, N, stds, minimalS_allowed, bkg_spec, channels_and_bins, central_values, output_file, buff_size, criterion, mu_bounds, seed, remove_channels, logger):
    set_seeds(seed)
    target_log_prob_fn = LikelihoodCalculatorWrapper(bkg_spec, channels_and_bins, central_values, output_file, buff_size, criterion, mu_bounds, seed, remove_channels, logger) 
    new_state_fn_truncated = NewStateWrapper(stds, minimalS_allowed)   
    RandomWalkMH=tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn, new_state_fn=new_state_fn_truncated, name=None)
    tfp.mcmc.sample_chain(
            num_results=int(N-1),
            num_burnin_steps=0,
            current_state=p0,
            kernel=RandomWalkMH,
            trace_fn=None,
            return_final_kernel_results = False,
            parallel_iterations=1,
            seed=seed,
            )
    final_counter = target_log_prob_fn.get_counter()
    if final_counter != 0:
        target_log_prob_fn.save_results(final_counter)
    return target_log_prob_fn.nLL_max

    

def calculate_sigmas(nSmin, nSmax, mask, SR_sigma, CR_sigma, VR_sigma, channels_and_bins):
    stds = np.empty(nSmax.shape, dtype=float)
    deltaS = nSmax-nSmin
    ii = 0
    for c, t, b in channels_and_bins:
        val = None
        if t=='SR':
            val = SR_sigma
        elif t=='CR':
            val = CR_sigma
        elif t=='VR':
            val = VR_sigma
        else:
            mes = f"[ERROR] Wrong type of the channel provided: {t}!"
            self.logger.critical(mes)
            raise ValueError(mes)
        stds[ii:ii+b] = val * deltaS[ii:ii+b]
        ii += b
    stds[~mask] = 0.0 # when signal leakage is disabled, do not vary CRs/VRs
    return stds


class ScanWrapper():
    def __init__(self, N,  bkg_spec, sigmas, channels_and_bins, central_values, buff_size, minimalS_allowed, criterion, mu_bounds, seed, remove_channels=None, logger=None):
        self._N = N 
        self._bkg_spec = bkg_spec
        self._channels_and_bins = channels_and_bins
        self._buff_size = buff_size
        self._minimalS_allowed = minimalS_allowed
        self._criterion = criterion
        self._stds = sigmas
        self._central_values = central_values
        self._mu_bounds = mu_bounds
        self._seed = seed
        self._remove_channels = remove_channels
        self.nLL_max = None
        if logger is None:
            self.logger = setup_logger()
        else:
            self.logger = logger

    
    def __call__(self, dat):
        set_seeds(self._seed)
        p0, output_file = dat
        if self._criterion in ['nLL_obs_mu1', 'nLL_exp_mu1', 'LL_obs_mu1', 'LL_exp_mu1']:
            criterion = self._criterion
        elif self._criterion == 'mu1':
            criterion = np.random.choice(['nLL_exp_mu1', 'nLL_obs_mu1'], 1)
        else:
            mes = '[ERROR] Wrong criterion passed to ScanWrapper.'
            self.logger.critical(mes)
            raise ValueError(mes)


        nLL_max = scan(p0, self._N, self._stds, self._minimalS_allowed, self._bkg_spec, \
            self._channels_and_bins, self._central_values, output_file, \
            self._buff_size, criterion, self._mu_bounds, self._seed, self._remove_channels, \
            self.logger)
        gc.collect()
        if self.nLL_max is None:
            self.nLL_max = nLL_max
            return self.nLL_max





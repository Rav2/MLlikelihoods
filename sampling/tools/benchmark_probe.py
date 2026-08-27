#!/usr/bin/env python3
#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file measures the cost of the lower-limit probe.
#

"""Time the find_min_S probe on a few bins and extrapolate to a full harvest.

The probe is the expensive part of a scan: for each bin it rebuilds the spey
model and evaluates two likelihoods, serially. Cost is therefore roughly

    total ~= setup + n_bins * t_bin

with t_bin dominated by the workspace size. This measures t_bin on a handful of
bins so a full harvest can be costed without running it.

Usage
-----
    python tools/benchmark_probe.py <analysis> [--patchset-index 0] [--bins 3]
                                    [--low-lim-samples 50] [--data-dir ../data]
"""
import argparse, json, os, sys, time
from collections import OrderedDict
import numpy as np

# the sampler modules live one level up, next to sample.py
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('analysis')
    ap.add_argument('--patchset-index', type=int, default=0)
    ap.add_argument('--bins', type=int, default=3, help='how many bins to actually probe')
    ap.add_argument('--low-lim-samples', type=int, default=50)
    ap.add_argument('--data-dir', default='../data')
    ap.add_argument('--card', default=None)
    args = ap.parse_args()

    import logging, yaml, spey, pyhf
    from misc import silence_spey_banner; silence_spey_banner()
    from spey_pyhf.helper_functions import WorkspaceInterpreter
    from utils import get_scan_limits, get_mask
    from likelihood import find_min_S

    log = logging.getLogger('bench')
    log.addHandler(logging.NullHandler())
    log.setLevel(logging.CRITICAL)

    card_path = args.card or f'cards/{args.analysis}.yaml'
    card = yaml.safe_load(open(card_path))
    i = args.patchset_index
    bkgs = card['bkgfiles']
    bkgfile = bkgs[i] if len(bkgs) > i else bkgs[0]
    ps = card['patchsets'][i]
    patchset = ps[0] if isinstance(ps, list) else ps
    channels = OrderedDict(card['channels'][i] if len(card['channels']) > i else card['channels'][0])
    adir = os.path.join(args.data_dir, args.analysis)

    print(f'analysis {args.analysis}  patchset[{i}] {patchset}  ({bkgfile})')
    print(f'spey {spey.__version__}, pyhf {pyhf.__version__}')

    t0 = time.perf_counter()
    bkg_spec = json.load(open(os.path.join(adir, bkgfile)))
    patch_json = json.load(open(os.path.join(adir, patchset)))
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    stat_wrapper = spey.get_backend('pyhf')
    interpreter = WorkspaceInterpreter(bkg_spec)
    smap, _ = interpreter.patch_to_map(signal_patch=patch_json['patches'][0]['patch'])
    for k, item in smap.items():
        interpreter.inject_signal(k, item)
    full = stat_wrapper(analysis=args.analysis, background_only_model=bkg_spec,
                        signal_patch=interpreter.make_patch())
    _, model_bkg, data_bkg = full.backend.model(expected=spey.ExpectationType.apriori)
    _, _, data_obs = full.backend.model(expected=spey.ExpectationType.observed)
    t_setup = time.perf_counter() - t0

    channels_and_bins, bins_names = [], []
    for name, n in model_bkg.config.channel_nbins.items():
        channels_and_bins.append((name, channels[name], n))
        for b in range(n):
            bins_names.append(f'{name}-{b}')
    nbins = len(bins_names)

    by = card.get('bkg_yields')
    bu = card.get('bkg_unc')
    if by and isinstance(by[0], list):
        by, bu = by[i], bu[i]
    if by and len(by) == nbins:
        order = [f'{c}-{b}' for c in channels for b in range(dict(model_bkg.config.channel_nbins)[c])]
        ymap, umap = dict(zip(order, by)), dict(zip(order, bu))
        bkg_yields = [(n, ymap[n]) for n in bins_names]
        bkg_unc = [(n, umap[n]) for n in bins_names]
    else:                       # fall back to the model's own expectation
        bkg_yields = list(zip(bins_names, [float(x) for x in data_bkg[:nbins]]))
        bkg_unc = [(n, float(np.sqrt(max(v, 1e-9)))) for n, v in bkg_yields]
        print('  (card has no per-bin yields for this patchset; using model expectation '
              'and sqrt(B) - timing is unaffected)')
    obs_yields = list(zip(bins_names, [float(x) for x in data_obs[:nbins]]))

    nSmin, nSmax, central = get_scan_limits(bkg_yields, bkg_unc, obs_yields, channels_and_bins,
                                            True, True, 0.10, 0.10, 'both', 'both',
                                            'obs', 'obs', log)
    del full, model_bkg

    k = min(args.bins, nbins)
    probe_mask = np.zeros(nbins, dtype=bool)
    probe_mask[:k] = True

    # match what sample.py does: single-threaded, deterministic TensorFlow.
    # Without this the benchmark quietly uses every core and under-reports.
    from likelihood import set_global_determinism
    set_global_determinism(seed=0)

    # find_min_S ACCUMULATES injected signal as it walks the bins, so bin N is
    # probed with negative signal already sitting in bins 0..N-1 and gets slower
    # as it goes. Time the first half and the second half of the sample
    # separately to expose that growth instead of extrapolating from bin 0.
    half = max(1, k // 2)
    mask_a = np.zeros(nbins, dtype=bool); mask_a[:half] = True
    t0 = time.perf_counter()
    find_min_S(args.low_lim_samples, bkg_spec, stat_wrapper, nSmin,
               channels_and_bins, log, probe_mask=mask_a, sig_rel_unc=0.0)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = find_min_S(args.low_lim_samples, bkg_spec, stat_wrapper, nSmin,
                     channels_and_bins, log, probe_mask=probe_mask, sig_rel_unc=0.0)
    t_probe = time.perf_counter() - t0

    t_bin = t_probe / k
    t_bin_early = t_first / half
    t_bin_late = (t_probe - t_first) / max(1, k - half)
    growth = t_bin_late / t_bin_early if t_bin_early else 1.0
    total = t_setup + t_bin * nbins
    print(f'\nper-bin cost, first {half} bins vs next {k-half}: '
          f'{t_bin_early:.1f} -> {t_bin_late:.1f} s/bin  (x{growth:.2f})')
    print(f'\nbins in this patchset : {nbins}')
    print(f'load json             : {t_load:8.1f} s')
    print(f'model setup           : {t_setup:8.1f} s')
    print(f'probe {k} bin(s)        : {t_probe:8.1f} s   -> {t_bin:.1f} s/bin')
    print(f'ESTIMATE full probe   : {total:8.1f} s  = {total/60:.1f} min  = {total/3600:.2f} h')
    print(f'\nfirst {k} probed limits (total yields): '
          f'{[round(float(central[j] + out[j]), 4) for j in range(k)]}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file reproduces post-fit background yields from a workspace.
#

"""Report background yields for an analysis: nominal, pre-fit and POST-FIT.

The nominal sum of a workspace's ``sample['data']`` is a PRE-FIT number. Papers
(and therefore the ``bkg_yields`` in an analysis card) normally quote the
POST-FIT background: the free normalisation factors - ``mu_Top``, ``mu_W``,
``mu_ST`` and friends - are fitted to the control regions first, which pulls
every sample they scale and changes the signal-region expectation too.

This prints, per channel and per bin:

  nominal        sum over samples of sample['data'], no modifiers applied
  post-fit       MLE fit to ALL regions, evaluated at the best-fit parameters
  CR-only fit    MLE fit to the CONTROL regions only, then extrapolated into
                 the signal regions - this is the standard ATLAS background-only
                 fit and the number a paper quotes
  observed       the recorded observation
  spey apriori / aposteriori   what spey returns for those ExpectationTypes

Which column matches a paper
----------------------------
The CR-only one. Fitting every region at once lets an excess in a signal region
pull the very normalisation factors that predict it, inflating the SR estimate;
on 1909.09226 the all-region fit gives SRHMEM = 12.67 where the CR-only fit and
the paper both give 8.1.

Note on spey's ExpectationType
------------------------------
``aposteriori`` does NOT give post-fit background yields: spey returns the
OBSERVED dataset for it, and the pre-fit expectation for ``apriori``. Neither
performs the background-only fit, which is why this tool does it with pyhf.

Usage
-----
    python tools/postfit_yields.py <analysis> [--data-dir ../data]
                                   [--bkgfile BkgOnly.json] [--patchset patchset.json]
                                   [--card cards/<analysis>.yaml]

With --card it also diffs against that card's bkg_yields.
"""
import argparse, json, os, sys
import numpy as np


def load_card_yields(card_path):
    import yaml
    c = yaml.safe_load(open(card_path))
    by = c.get('bkg_yields')
    if not by:
        return None, None
    if isinstance(by[0], list):
        by = by[0]
    group = c['channels'][0] if isinstance(c['channels'], list) else c['channels']
    return list(group.keys()), by



def cr_only_fit_with_uncertainty(ws, model, sr_channels):
    """CR-only fit + full post-fit covariance propagated into the yields.

    Mirrors the ATLAS background-only fit: the control regions alone constrain
    the normalisation factors, the signal regions are extrapolated, and a
    parameter the CR fit cannot see (an SR's own MC-statistics term) keeps its
    PRIOR width rather than being treated as exactly known.

    Returns (yields, covariance, fitted_values, fitted_errors, model_cr) or
    (None,)*5 if the fit fails.
    """
    import pyhf, numpy as np
    # spey switches pyhf to the jax backend when it builds its model; minuit is
    # what supplies parameter uncertainties, so pin the backend here rather than
    # once at import time
    pyhf.set_backend('numpy', 'minuit')
    try:
        ws_cr = ws.prune(channels=sr_channels)
        model_cr = ws_cr.model(poi_name=None)
        best, corr = pyhf.infer.mle.fit(ws_cr.data(model_cr), model_cr,
                                        return_correlations=True, return_uncertainties=True)
    except Exception as e:
        print(f'  [CR-only fit failed: {e!r}]')
        return (None,) * 5
    vals, errs = best[:, 0], best[:, 1]
    cov_cr = corr * np.outer(errs, errs)

    npar = len(model.config.suggested_init())
    pars = np.array(model.config.suggested_init(), dtype=float)
    cov = np.zeros((npar, npar))
    idx = {}
    for pname, pspec in model_cr.config.par_map.items():
        if pname not in model.config.par_map:
            continue
        s_cr, s_full = pspec['slice'], model.config.par_map[pname]['slice']
        if (s_cr.stop - s_cr.start) != (s_full.stop - s_full.start):
            continue
        pars[s_full] = vals[s_cr]
        for k in range(s_full.stop - s_full.start):
            idx[s_full.start + k] = s_cr.start + k
    for fi, ci in idx.items():
        for fj, cj in idx.items():
            cov[fi, fj] = cov_cr[ci, cj]
    for pname, pspec in model.config.par_map.items():
        if pname in model_cr.config.par_map:
            continue
        pset = model.config.param_set(pname)
        w = np.asarray(pset.width(), dtype=float) if pset.constrained else None
        s = pspec['slice']
        for k in range(s.stop - s.start):
            cov[s.start + k, s.start + k] = float(w[k]) ** 2 if w is not None else 0.0

    mu = np.asarray(model.expected_actualdata(list(pars)), dtype=float)
    J = np.zeros((len(mu), npar))
    for i in range(npar):
        h = max(1e-5, abs(pars[i]) * 1e-4)
        up, dn = pars.copy(), pars.copy()
        up[i] += h; dn[i] -= h
        J[:, i] = (np.asarray(model.expected_actualdata(list(up)), dtype=float)
                   - np.asarray(model.expected_actualdata(list(dn)), dtype=float)) / (2 * h)
    return mu, J @ cov @ J.T, vals, errs, model_cr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('analysis')
    ap.add_argument('--data-dir', default='../data')
    ap.add_argument('--bkgfile', default=None, help='defaults to the card\'s first bkgfile')
    ap.add_argument('--patchset', default=None, help='defaults to the card\'s first patchset')
    ap.add_argument('--card', default=None)
    args = ap.parse_args()

    import pyhf, spey
    from misc import silence_spey_banner; silence_spey_banner()
    pyhf.set_backend('numpy', 'minuit')
    from spey_pyhf.helper_functions import WorkspaceInterpreter

    adir = os.path.join(args.data_dir, args.analysis)
    card_path = args.card or f'cards/{args.analysis}.yaml'
    bkgfile, patchset = args.bkgfile, args.patchset
    if (bkgfile is None or patchset is None) and os.path.isfile(card_path):
        import yaml
        c = yaml.safe_load(open(card_path))
        bkgfile = bkgfile or c['bkgfiles'][0]
        ps = c['patchsets'][0]
        patchset = patchset or (ps[0] if isinstance(ps, list) else ps)
    bkgfile = bkgfile or 'BkgOnly.json'

    spec = json.load(open(os.path.join(adir, bkgfile)))
    print(f'analysis   : {args.analysis}')
    print(f'background : {bkgfile}')

    # ---------------------------------------------------------------- nominal
    nominal = {}
    for ch in spec['channels']:
        n = len(ch['samples'][0]['data'])
        acc = [0.0] * n
        for s in ch['samples']:
            for i, v in enumerate(s['data']):
                acc[i] += v
        nominal[ch['name']] = acc
    observed = {o['name']: list(o['data']) for o in spec['observations']}

    # ------------------------------------------------- spey apriori/aposteriori
    spey_data = {}
    if patchset and os.path.isfile(os.path.join(adir, patchset)):
        print(f'patchset   : {patchset}')
        patch_json = json.load(open(os.path.join(adir, patchset)))
        signal_patch = patch_json['patches'][0]['patch']
        interpreter = WorkspaceInterpreter(spec)
        smap, _ = interpreter.patch_to_map(signal_patch=signal_patch)
        for k, item in smap.items():
            interpreter.inject_signal(k, item)
        stat_wrapper = spey.get_backend('pyhf')
        stat_model = stat_wrapper(analysis=args.analysis,
                                  background_only_model=spec,
                                  signal_patch=interpreter.make_patch())
        for label, etype in (('apriori', spey.ExpectationType.apriori),
                             ('aposteriori', spey.ExpectationType.aposteriori),
                             ('observed', spey.ExpectationType.observed)):
            try:
                _, mdl, dat = stat_model.backend.model(expected=etype)
                spey_data[label] = (np.asarray(dat, dtype=float), dict(mdl.config.channel_nbins))
            except Exception as e:
                print(f'  [spey {label} unavailable: {e!r}]')

    # ------------------------------------------------------------- post-fit MLE
    ws = pyhf.Workspace(spec)
    model = ws.model(poi_name=None)
    data = ws.data(model)
    init = model.config.suggested_init()
    prefit = np.asarray(model.expected_actualdata(init), dtype=float)
    try:
        bestfit = pyhf.infer.mle.fit(data, model)
        postfit = np.asarray(model.expected_actualdata(bestfit), dtype=float)
    except Exception as e:
        print(f'  [post-fit failed: {e!r}]')
        bestfit, postfit = None, None

    # ------------------------------------------- CR-only ("background-only") fit
    # The standard ATLAS procedure fits the control regions ALONE and then
    # extrapolates into the signal regions, so an excess in an SR cannot pull the
    # normalisation factors that predict it. Fitting everything at once (above)
    # lets the SRs drag their own prediction upwards.
    postfit_cr, cov_cr_yield, sr_channels = None, None, []
    if os.path.isfile(card_path):
        import yaml
        c = yaml.safe_load(open(card_path))
        group = c['channels'][0] if isinstance(c['channels'], list) else c['channels']
        sr_channels = [ch for ch, kind in group.items() if kind == 'SR']
    bestfit_cr = errs_cr = model_cr = None
    if sr_channels:
        postfit_cr, cov_cr_yield, bestfit_cr, errs_cr, model_cr = \
            cr_only_fit_with_uncertainty(ws, model, sr_channels)

    if bestfit is not None:
        print('\nfitted normalisation factors  (all-region fit -> CR-only fit):')
        for name, spec_par in model.config.par_map.items():
            mod = spec_par.get('paramset')
            if getattr(mod, 'is_shared', None) is None and not hasattr(mod, 'n_parameters'):
                continue
            sl = spec_par['slice']
            if sl.stop - sl.start == 1 and type(mod).__name__ == 'unconstrained':
                extra = ''
                if bestfit_cr is not None and model_cr is not None and name in model_cr.config.par_map:
                    scr = model_cr.config.par_map[name]['slice']
                    extra = f'   CR-only: {bestfit_cr[scr.start]:6.3f} +/- {errs_cr[scr.start]:.3f}'
                print(f'   {name:12s} {init[sl.start]:6.3f}  ->  {bestfit[sl.start]:6.3f}{extra}')

    # ---------------------------------------------------------------- report
    card_names, card_vals = (None, None)
    if os.path.isfile(card_path):
        card_names, card_vals = load_card_yields(card_path)

    nbins = dict(model.config.channel_nbins)
    hdr = f'{"channel":22s} {"nominal":>10s} {"post-fit":>10s} {"CR-only fit":>12s} {"observed":>9s}'
    for lab in ('apriori', 'aposteriori'):
        if lab in spey_data:
            hdr += f' {"spey " + lab[:6]:>12s}'
    print('\n' + hdr)
    print('-' * len(hdr))
    i = 0
    tot = dict(nominal=0.0, prefit=0.0, postfit=0.0, obs=0.0)
    for name, n in nbins.items():
        nom = sum(nominal[name])
        pre = float(prefit[i:i + n].sum())
        post = float(postfit[i:i + n].sum()) if postfit is not None else float('nan')
        obs = sum(observed[name])
        pcr = float(postfit_cr[i:i + n].sum()) if postfit_cr is not None else float('nan')
        row = f'{name:22s} {nom:10.2f} {post:10.2f} {pcr:12.2f} {obs:9.0f}'
        for lab in ('apriori', 'aposteriori'):
            if lab in spey_data:
                d, _ = spey_data[lab]
                row += f' {float(d[i:i + n].sum()):12.2f}'
        print(row)
        tot['nominal'] += nom; tot['prefit'] += pre; tot['postfit'] += post; tot['obs'] += obs
        i += n
    print('-' * len(hdr))
    tcr = float(postfit_cr.sum()) if postfit_cr is not None else float('nan')
    print(f'{"TOTAL":22s} {tot["nominal"]:10.2f} {tot["postfit"]:10.2f} '
          f'{tcr:12.2f} {tot["obs"]:9.0f}')

    # per-bin post-fit, and the card comparison
    if postfit_cr is not None:
        print('\nper-bin CR-only-fit background (the number a paper quotes):')
        i = 0
        for name, n in nbins.items():
            if cov_cr_yield is not None:
                errs = [float(np.sqrt(max(cov_cr_yield[j, j], 0))) for j in range(i, i + n)]
                vals = ', '.join(f'{v:.2f}+/-{e:.2f}' for v, e in zip(postfit_cr[i:i + n], errs))
            else:
                vals = ', '.join(f'{v:.2f}' for v in postfit_cr[i:i + n])
            print(f'   {name:22s} [{vals}]')
            i += n

    if card_vals:
        print(f'\ncard {os.path.basename(card_path)} lists {len(card_vals)} bkg_yields for '
              f'{sum(nbins.values())} model bins across {len(nbins)} channels')
        print(f'   card channel order : {card_names}')
        print(f'   card values        : {card_vals}')
        if len(card_vals) == len(card_names):
            print(f'\n   {"channel":22s} {"card":>9s} {"CR-only":>10s} {"nominal":>9s} {"observed":>9s}')
            i = 0
            offsets = {}
            for name, n in nbins.items():
                offsets[name] = (i, n); i += n
            for ch, cv in zip(card_names, card_vals):
                if ch not in offsets:
                    continue
                o, n = offsets[ch]
                post = float(postfit_cr[o:o + n].sum()) if postfit_cr is not None else float(postfit[o:o + n].sum())
                nom = sum(nominal[ch])
                obs = sum(observed[ch])
                print(f'   {ch:22s} {cv:9.2f} {post:10.2f} {nom:9.2f} {obs:9.0f}')


if __name__ == '__main__':
    main()

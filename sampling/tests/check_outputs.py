#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file checks the outputs a scan produced.
#

"""Validate the CSV/JSON produced by a sampling run."""
import json, os, sys, glob
import numpy as np
import pandas as pd

root = sys.argv[1]
ok = True

for d in sorted(glob.glob(os.path.join(root, '*-*'))):
    csvs = glob.glob(os.path.join(d, 'results-*.csv'))
    if not csvs:
        continue
    csv = csvs[0]
    meta = json.load(open(csv.replace('.csv', '.json')))
    df = pd.read_csv(csv)
    name = os.path.basename(d)
    print(f'\n=== {name} ===')

    n_lik = 8
    bin_cols = list(df.columns[:-n_lik])
    lik_cols = list(df.columns[-n_lik:])
    expected_rows = meta['scans'] * meta['points']

    def report(label, cond, detail=''):
        global ok
        print(f'  [{"OK " if cond else "FAIL"}] {label}{(" - " + detail) if detail and not cond else ""}')
        if not cond:
            ok = False

    report(f'row count == scans*points ({expected_rows})', len(df) == expected_rows, f'got {len(df)}')

    removed = meta.get('remove_channels') or []
    all_bins = [b for b, _ in meta['bkg_yields']]
    kept = [b for b in all_bins if b.rsplit('-', 1)[0] not in removed]
    report(f'{len(bin_cols)} bin columns, removed={removed}', bin_cols == sorted(kept),
           f'cols={bin_cols} expected={sorted(kept)}')

    report('no NaN anywhere', not df.isnull().values.any())
    report('all likelihoods finite', bool(np.isfinite(df[lik_cols].values).all()))
    report('no 1e10 sentinel (NaN substitute)', not bool((df[lik_cols].abs() >= 1e9).any().any()))

    # yields must respect the scan limits recorded in the metadata
    lower = {b: v for (b, _), v in zip(meta['bkg_yields'], meta['lower_limits'])}
    upper = {b: v for (b, _), v in zip(meta['bkg_yields'], meta['upper_limits'])}
    viol = []
    for c in bin_cols:
        lo, hi = lower[c], upper[c]
        # MCMC proposals are truncated from below only; the upper edge can be
        # exceeded by a proposal, so only the hard lower bound is checked
        if (df[c] < lo - 1e-6).any():
            viol.append(f'{c} < {lo} (min {df[c].min()})')
    report('yields respect the lower scan limits', not viol, str(viol[:3]))

    # A pinned bin (signal leakage off for its region) is never varied, so it has
    # to sit EXACTLY on its central value in every row. The +1e-4 probe offset in
    # find_min_S used to break this before pinned bins were skipped there.
    leak = {'CR': meta['signal_leakage_CR'], 'VR': meta['signal_leakage_VR']}
    pinned = [c for c in bin_cols
              if c[:2] in leak and not leak[c[:2]]]
    if pinned:
        bad = {c: [float(v) for v in sorted(df[c].unique())[:3]] for c in pinned if df[c].nunique() != 1}
        report(f'{len(pinned)} pinned bins are exactly constant', not bad, str(bad))

    report('mu0 likelihood constant within a scan',
           all(df[c].nunique() <= meta['scans'] + 1 for c in ['nLL_exp_mu0', 'nLL_obs_mu0']))
    report('metadata has max-likelihood values', len(meta.get('nLL_obs_max') or []) > 0)
    report('starting points NOT stored in metadata', 'starting_points' not in meta)
    # the placeholder marks a likelihood that came back NaN/inf; merge_results
    # drops those rows, so a finished file must not contain any
    report('no NaN placeholder survived the merge',
           not bool((df[lik_cols].abs() >= 1e10 * (1 - 1e-9)).any().any()))

    print(f'  info: criterion={meta["scan_criterion"]}, start={meta["start_method"]}, '
          f'fit_bkg={meta["fit_bkg"]}, sig_rel_unc={meta["sig_rel_unc"]}')
    print(f'  info: nLL_obs_mu1 range [{df["nLL_obs_mu1"].min():.4g}, {df["nLL_obs_mu1"].max():.4g}]')

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)

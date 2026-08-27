#!/usr/bin/env python3
#
# author: Rafal Maselek
# e-mail: rafal.maselek@ijs.si
# ORCID:  https://orcid.org/0000-0002-5558-8249
#
# This file compares reproduced yields against HEPData.
#

"""Validate the CR-only background fit against HEPData post-fit tables.

Parses the "Expected background" block out of HEPData's dataMC_SR*_mct CSV
exports and compares it, bin by bin, with what
``postfit_yields.cr_only_fit_with_uncertainty`` produces from the published
likelihood. This is the end-to-end check that the reimplemented ATLAS
background-only fit really reproduces the experiment.

Usage
-----
    python tools/validate_vs_hepdata.py 1909.09226 csv_files/
"""
import csv, os, re, sys
import numpy as np


def parse_hepdata_csv(path):
    """Return (bin_edges, expected, err_up, err_dn, observed) from a HEPData export.

    The file holds several blocks separated by blank lines; each starts with a
    header row whose 4th column names the quantity. We want the block whose
    name begins with 'Expected background', and the one for 'Data'.
    """
    blocks, cur = [], []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip():
            if cur:
                blocks.append(cur); cur = []
            continue
        if line.startswith('#:'):
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)

    expected = observed = None
    for b in blocks:
        rows = list(csv.reader(b))
        if len(rows) < 2:
            continue
        header = rows[0]
        if len(header) < 4:
            continue
        label = header[3]
        vals = []
        for r in rows[1:]:
            try:
                lo, hi, v = float(r[1]), float(r[2]), float(r[3])
            except (ValueError, IndexError):
                continue
            up = float(r[4]) if len(r) > 4 and r[4] not in ('', None) else None
            dn = float(r[5]) if len(r) > 5 and r[5] not in ('', None) else None
            vals.append((lo, hi, v, up, dn))
        # HEPData is inconsistent here: SR-HM says "Expected background" while
        # SR-LM and SR-MM say "Expected Bakground" (sic, typo in the record)
        if re.match(r'\s*expected\s+ba[kc]?k?ground', label, re.I):
            expected = vals
        elif re.match(r'\s*data\b', label, re.I):
            observed = vals
    return expected, observed


def main():
    analysis = sys.argv[1] if len(sys.argv) > 1 else '1909.09226'
    csv_dir = sys.argv[2] if len(sys.argv) > 2 else 'csv_files'
    data_dir = sys.argv[3] if len(sys.argv) > 3 else '../data'

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from postfit_yields import cr_only_fit_with_uncertainty
    import pyhf, yaml, json

    card = yaml.safe_load(open(f'cards/{analysis}.yaml'))
    spec = json.load(open(os.path.join(data_dir, analysis, card['bkgfiles'][0])))
    group = card['channels'][0] if isinstance(card['channels'], list) else card['channels']
    sr_channels = [ch for ch, k in group.items() if k == 'SR']

    ws = pyhf.Workspace(spec)
    model = ws.model(poi_name=None)
    mu, cov, _, _, _ = cr_only_fit_with_uncertainty(ws, model, sr_channels)
    if mu is None:
        raise SystemExit('fit failed')
    nbins = dict(model.config.channel_nbins)

    # map HEPData table name -> workspace channel
    files = {}
    for f in sorted(os.listdir(csv_dir)):
        m = re.search(r'dataMC_(SR[A-Z]{2})_mct\.csv$', f)
        if m:
            files[m.group(1)] = os.path.join(csv_dir, f)
    # SRHM -> SRHMEM_mct2 etc.
    chan_of = {}
    for tag in files:
        for ch in sr_channels:
            if ch.upper().startswith(tag):
                chan_of[tag] = ch
    if not chan_of:
        raise SystemExit(f'could not match {list(files)} to channels {sr_channels}')

    offsets, i = {}, 0
    for name, n in nbins.items():
        offsets[name] = (i, n); i += n

    print(f'{"region / bin":26s} {"mCT range":>14s} {"HEPData":>16s} {"this fit":>16s} '
          f'{"d(val)":>8s} {"d(err)":>8s}')
    print('-' * 96)
    worst_v = worst_e = 0.0
    n_cmp = 0
    for tag in sorted(chan_of):
        ch = chan_of[tag]
        exp, obs = parse_hepdata_csv(files[tag])
        off, n = offsets[ch]
        if exp is None:
            print(f'{tag}: no "Expected background" block found'); continue
        if len(exp) != n:
            print(f'{tag}: HEPData has {len(exp)} bins, workspace has {n} - comparing the first {min(len(exp),n)}')
        tot_h = tot_f = 0.0
        for b in range(min(len(exp), n)):
            lo, hi, v, up, dn = exp[b]
            f_v = float(mu[off + b])
            f_e = float(np.sqrt(max(cov[off + b, off + b], 0)))
            dv = (f_v - v) / v * 100 if v else float('nan')
            de = (f_e - abs(up)) / abs(up) * 100 if up else float('nan')
            worst_v = max(worst_v, abs(dv)); worst_e = max(worst_e, abs(de)); n_cmp += 1
            tot_h += v; tot_f += f_v
            print(f'{ch + "-" + str(b):26s} {f"{lo:.0f}-{hi:.0f}":>14s} '
                  f'{f"{v:.3f} +/- {abs(up):.3f}":>16s} {f"{f_v:.3f} +/- {f_e:.3f}":>16s} '
                  f'{dv:+7.2f}% {de:+7.2f}%')
        sl = slice(off, off + n)
        tot_e = float(np.sqrt(max(cov[sl, sl].sum(), 0)))
        print(f'{"  " + ch + " TOTAL":26s} {"":>14s} {tot_h:16.3f} '
              f'{f"{tot_f:.3f} +/- {tot_e:.3f}":>16s} {(tot_f-tot_h)/tot_h*100:+7.2f}%')
        print()
    print('-' * 96)
    print(f'{n_cmp} bins compared;  largest deviation: central {worst_v:.2f}%, '
          f'uncertainty {worst_e:.2f}%')


if __name__ == '__main__':
    main()

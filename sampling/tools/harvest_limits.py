#!/usr/bin/env python3
"""Harvest scan limits for an analysis and write them into its card.

The lower-limit probe (``find_min_S``) is by far the most expensive part of a
scan - hours on a large workspace - and it produces the same numbers every
time for a given background model and signal uncertainty. This tool runs it
once and stores the result in the card as ``scan_limits``, so later scans load
the limits instead of recomputing them.

Two steps:

  prepare   build the parameter files needed to compute the limits, one per
            DISTINCT background model in the card (patchsets sharing a
            background file and channel group share their limits, so they are
            only computed once)

  merge     read the metadata JSON produced by those runs and write the
            ``scan_limits`` block into the card

Example
-------
    python tools/harvest_limits.py prepare cards/1908.08215.yaml 1908.08215 \
        --outdir harvest --sig-rel-unc 0.0
    # run each generated parameter file (locally or via SLURM), then:
    python tools/harvest_limits.py merge cards/1908.08215.yaml \
        harvest/out/*/metadata.json --sig-rel-unc 0.0

Limits are stored as TOTAL yields (central value +/- the scan range), matching
what the sampler prints and what it stores in the run metadata.

NOTE: limits depend on ``sig_rel_unc``, because the probe injects the same
signal-uncertainty modifier the scan uses. Harvest separately for each value.
"""
import argparse, json, os, sys
import yaml


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def patchset_names(card):
    return [p[0] if isinstance(p, list) else p for p in card['patchsets']]


def channel_group(card, index):
    return card['channels'][index] if len(card['channels']) > index else card['channels'][0]


def background_key(card, index):
    """Patchsets with the same key share a background model, hence their limits."""
    bkgs = card['bkgfiles']
    bkg = bkgs[index] if len(bkgs) > index else bkgs[0]
    return (bkg, tuple(channel_group(card, index).items()))


def distinct_background_groups(card):
    """Map background key -> list of patchset indices sharing it."""
    groups = {}
    for i in range(len(card['patchsets'])):
        groups.setdefault(background_key(card, i), []).append(i)
    return groups


def card_bin_names(card, index, nbins_by_channel):
    names = []
    for ch in channel_group(card, index):
        for b in range(nbins_by_channel.get(ch, 1)):
            names.append(f'{ch}-{b}')
    return names


def workspace_nbins(data_dir, bkgfile):
    ws = json.load(open(os.path.join(data_dir, bkgfile)))
    return {ch['name']: len(ch['samples'][0]['data']) for ch in ws['channels']}


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------
def cmd_prepare(args):
    card_text = open(args.card).read()
    card = yaml.safe_load(card_text)
    names = patchset_names(card)
    os.makedirs(args.outdir, exist_ok=True)

    groups = distinct_background_groups(card)
    print(f'{len(card["patchsets"])} patchsets -> {len(groups)} distinct background model(s) to probe')

    generated = []
    for key, indices in groups.items():
        lead = indices[0]
        tag = f'{args.analysis}-ps{lead}-sig{args.sig_rel_unc}'.replace('.', '_')
        c = yaml.safe_load(card_text)
        c.pop('scan_limits', None)                 # force the COMPUTED path
        c['sig_rel_unc'] = args.sig_rel_unc
        c['patchsets'] = [[n, i == lead] for i, n in enumerate(names)]
        card_path = os.path.join(args.outdir, f'card-{tag}.yaml')
        with open(card_path, 'w') as f:
            yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False, width=200)

        # Harvest the GENERAL box: leakage on in every region and nothing removed,
        # so every bin is probed. A later run re-imposes its own pinning and
        # removal on top of the stored limits; a box harvested with regions
        # already pinned or channels already dropped would not be reusable.
        harvest_cfg = {
            'signal_leakage_CR': True,
            'signal_leakage_VR': True,
            'signal_leakage_CR_spread': args.cr_spread,
            'signal_leakage_VR_spread': args.vr_spread,
            'signal_leakage_CR_sign': args.sign,
            'signal_leakage_VR_sign': args.sign,
            'CR_center': args.cr_center,
            'VR_center': args.vr_center,
            'remove_channels': [],
            'removeCRsVRs': False,
        }
        glob_doc = {
            'analyses': [args.analysis],
            'input_folder': args.input_folder,
            'output_folder': os.path.join(args.out_root, tag) + '/',
            'processes': 1, 'scans': 1, 'points': args.points,
            'buffer_size': max(5, args.points), 'low_lim_samples': args.low_lim_samples,
            'cluster': False, 'spey_verbose_lvl': 0, 'keep_files': True,
        }
        ana_doc = {
            'analysis': args.analysis,
            'include': os.path.relpath(card_path, args.rel_to),
            'scans': 1, 'processes': 1, 'points': args.points,
            'buffer_size': max(5, args.points),
            'low_lim_samples': args.low_lim_samples,
            'start_method': 'random',
            **harvest_cfg,
        }
        # the card is merged ON TOP of the analysis document, so the harvest
        # settings have to live in the card too or they would be overridden
        c.update(harvest_cfg)
        with open(card_path, 'w') as f:
            yaml.safe_dump(c, f, default_flow_style=False, sort_keys=False, width=200)
        par_path = os.path.join(args.outdir, f'params-{tag}.yaml')
        with open(par_path, 'w') as f:
            yaml.safe_dump_all([glob_doc, ana_doc], f, default_flow_style=False, sort_keys=False)
        generated.append(par_path)
        covered = ', '.join(names[i] for i in indices)
        print(f'  {os.path.basename(par_path)}')
        print(f'      probes {names[lead]}; result also applies to: {covered}')
    print('\nRun each parameter file, then merge the metadata JSON it produces.')
    return generated


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------
def cmd_merge(args):
    card_text = open(args.card).read()
    card = yaml.safe_load(card_text)
    names = patchset_names(card)
    groups = distinct_background_groups(card)

    # index metadata by the set of bin names it covers
    metas = []
    for p in args.metadata:
        m = json.load(open(p))
        if m.get('analysis') != args.analysis:
            print(f'  skipping {p}: analysis is {m.get("analysis")!r}, expected {args.analysis!r}')
            continue
        if args.sig_rel_unc is not None and abs(float(m.get('sig_rel_unc', 0.0)) - args.sig_rel_unc) > 1e-12:
            raise SystemExit(f'{p}: harvested at sig_rel_unc={m.get("sig_rel_unc")} but '
                             f'--sig-rel-unc {args.sig_rel_unc} was requested. Limits depend on it.')
        metas.append((p, m))
    if not metas:
        raise SystemExit('no usable metadata files')

    nbins = {}
    for i in range(len(card['patchsets'])):
        bkgs = card['bkgfiles']
        bkg = bkgs[i] if len(bkgs) > i else bkgs[0]
        nbins[i] = workspace_nbins(os.path.join(args.data_dir, args.analysis), bkg)

    entries = [None] * len(card['patchsets'])
    provenance = {}
    for key, indices in groups.items():
        wanted = set(card_bin_names(card, indices[0], nbins[indices[0]]))
        match = None
        for p, m in metas:
            if set(b for b, _ in m['bkg_yields']) == wanted:
                match = (p, m); break
        if match is None:
            print(f'  no metadata covering {names[indices[0]]} ({len(wanted)} bins) - left as null')
            continue
        p, m = match
        by = {b: (lo, hi) for (b, _), lo, hi in zip(m['bkg_yields'], m['lower_limits'], m['upper_limits'])}
        for i in indices:
            order = card_bin_names(card, i, nbins[i])
            entries[i] = [[by[b][0], by[b][1]] for b in order]
            provenance[i] = (os.path.basename(p), m.get('seed'))
        print(f'  {os.path.basename(p)} -> patchsets {indices} ({len(wanted)} bins)')

    context = {
        'sig_rel_unc': args.sig_rel_unc,
        'signal_leakage_CR_spread': metas[0][1].get('signal_leakage_CR_spread'),
        'signal_leakage_VR_spread': metas[0][1].get('signal_leakage_VR_spread'),
        'CR_center': metas[0][1].get('CR_center'),
        'VR_center': metas[0][1].get('VR_center'),
    }
    block = render_block(card, entries, names, nbins, provenance, args.sig_rel_unc)
    out = args.output or args.card
    write_card(card_text, out, block, args.sig_rel_unc, context)
    print(f'\nwrote {out}')


def render_block(card, entries, names, nbins, provenance, sig_rel_unc):
    lines = ['scan_limits :']
    for i, entry in enumerate(entries):
        if entry is None:
            lines.append(f'    # {names[i]}: not hardcoded - computed at run time')
            lines.append('    - null')
            continue
        src, seed = provenance.get(i, ('?', '?'))
        lines.append(f'    # {names[i]}: {len(entry)} bins, same channel order as "channels" above')
        lines.append(f'    #   source: {src} (seed {seed})')
        lines.append('    -')
        for (lo, hi), b in zip(entry, card_bin_names(card, i, nbins[i])):
            lines.append(f'        - [{lo:.10g}, {hi:.10g}]   # {b}')
    return '\n'.join(lines) + '\n'


MARKER = '\n#\n# --- scan limits'


def write_card(card_text, out_path, block, sig_rel_unc, context=None):
    text = card_text.rstrip('\n')
    if MARKER in text:
        text = text[:text.index(MARKER)].rstrip('\n')
    header = (
        f'(TOTAL yields: central value +/- the scan range, i.e. exactly what\n'
        f'# the sampler prints in its "Scan limits" table and stores as\n'
        f'# lower_limits / upper_limits in the run metadata). One entry per\n'
        f'# patchset, in the same order as "patchsets"; null means "compute at\n'
        f'# run time".\n'
        f'#\n'
        f'# Harvested at sig_rel_unc = {sig_rel_unc}. The limits DEPEND on that value,\n'
        f'# because the probe injects the same signal-uncertainty modifier the scan\n'
        f'# uses - do not reuse this block for a different signal uncertainty.\n'
        f'#\n'
        f'# Harvested over ALL bins, with signal leakage enabled everywhere and\n'
        f'# nothing removed, so the box stays valid whatever a later run pins or\n'
        f'# drops; the sampler re-imposes that run\'s pinning on load.\n'
        f'#\n'
        f'# Harvest context: ' + ', '.join(f'{k}={v}' for k, v in (context or {}).items()) + '\n'
        f'#\n'
        f'# Generated by tools/harvest_limits.py so the expensive find_min_S probe\n'
        f'# does not have to run again.'
    )
    with open(out_path, 'w') as f:
        f.write(text + MARKER + ' ' + header + '\n#\n' + block)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('prepare', help='write the parameter files needed to compute the limits')
    p.add_argument('card'); p.add_argument('analysis')
    p.add_argument('--outdir', default='harvest')
    p.add_argument('--out-root', default='/tmp/harvest')
    p.add_argument('--input-folder', default='../data/')
    p.add_argument('--rel-to', default='.', help='directory sample.py will be run from; include paths are made relative to it')
    p.add_argument('--sig-rel-unc', type=float, default=0.0)
    p.add_argument('--cr-spread', type=float, default=0.10,
                   help='signal_leakage_CR_spread to harvest with; it is baked into the CR limits')
    p.add_argument('--vr-spread', type=float, default=0.10,
                   help='signal_leakage_VR_spread to harvest with; it is baked into the VR limits')
    p.add_argument('--sign', default='both', choices=['both', 'positive', 'negative'])
    p.add_argument('--cr-center', default='obs', choices=['obs', 'exp'])
    p.add_argument('--vr-center', default='obs', choices=['obs', 'exp'])
    p.add_argument('--points', type=int, default=2)
    p.add_argument('--low-lim-samples', type=int, default=50)
    p.set_defaults(func=cmd_prepare)

    m = sub.add_parser('merge', help='write harvested limits into the card')
    m.add_argument('card'); m.add_argument('metadata', nargs='+')
    m.add_argument('--analysis', required=True)
    m.add_argument('--data-dir', default='../data')
    m.add_argument('--sig-rel-unc', type=float, default=None)
    m.add_argument('--output', default=None, help='write here instead of overwriting the card')
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

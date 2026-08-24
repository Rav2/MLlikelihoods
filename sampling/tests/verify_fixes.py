"""Checks for the sampling bug-fix pass.

Each test targets one of the reported bugs and FAILS on the original code.
Run from inside the `sampling/` directory:  python3 ../verify_fixes.py
"""
import logging, sys, os, tempfile
import numpy as np

sys.path.insert(0, os.getcwd())  # run from inside sampling/

logging.basicConfig(level=logging.CRITICAL)
log = logging.getLogger('verify')
log.addHandler(logging.NullHandler())

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((True, name, ''))
        print(f'  PASS  {name}')
    except Exception as e:
        RESULTS.append((False, name, repr(e)))
        print(f'  FAIL  {name}\n          {e!r}')


# ---------------------------------------------------------------- bug 1
def test_mask_multibin():
    """Channel-removal mask must switch off EVERY bin of a removed channel."""
    import likelihood
    cab = [('CR_a', 'CR', 3), ('SR_x', 'SR', 2), ('SR_y', 'SR', 4)]
    total = sum(b for _, _, b in cab)

    # replicate __init__'s bookkeeping without building a pyhf model
    remove = ['CR_a', 'SR_y']
    bin_no = total
    mask = np.ones(shape=bin_no, dtype=bool)
    bin_offset = 0
    for c, sr, b in cab:
        if c in remove:
            bin_no -= b
            mask[bin_offset:bin_offset + b] = False
        bin_offset += b
    assert bin_no == 2, f'expected 2 surviving bins, got {bin_no}'
    assert int(mask.sum()) == bin_no, f'mask keeps {mask.sum()} bins, bookkeeping says {bin_no}'
    assert list(mask) == [False]*3 + [True]*2 + [False]*4, list(mask)

    # and the shipped source must actually contain the slice assignment
    src = open('likelihood.py').read()
    assert 'self._mask[bin_offset:bin_offset + b] = False' in src, 'slice assignment missing from likelihood.py'
    assert 'Bin bookkeeping mismatch' in src, 'consistency guard missing from likelihood.py'


# ---------------------------------------------------------------- bug 3
def test_populate_randomly_signature():
    """start_method 'default' with scans > 3 must not raise TypeError."""
    import utils
    nsMin = np.array([-5.0, -3.0, -1e-10, -1e-10])
    nsMax = np.array([10.0, 8.0, 1e-10, 1e-10])
    central = np.array([20.0, 15.0, 100.0, 50.0])
    mask = np.array([True, True, False, False])
    pts = utils.generate_starting_points(nsMin, nsMax, central, mask=mask, n=6,
                                         start_method='default',
                                         channels_and_bins=[('SR', 'SR', 2), ('CR', 'CR', 2)],
                                         logger=log)
    assert pts.shape == (6, 4), pts.shape
    assert np.all(pts[:, ~mask] == 0.0), 'unmasked bins must stay at 0'


def test_edges_topup():
    """start_method 'edges' with more points than edges must top up, not crash."""
    import utils
    nsMin = np.array([-1.0, -2.0])
    nsMax = np.array([1.0, 2.0])
    central = np.array([5.0, 6.0])
    mask = np.array([True, True])
    pts = utils.generate_starting_points(nsMin, nsMax, central, mask=mask, n=10,
                                         start_method='edges',
                                         channels_and_bins=[('SR', 'SR', 2)], logger=log)
    assert pts.shape == (10, 2), pts.shape


def test_bad_start_method_falls_back():
    """An unknown start_method must fall back instead of raising NameError."""
    import utils
    nsMin = np.array([-1.0, -2.0]); nsMax = np.array([1.0, 2.0])
    central = np.array([5.0, 6.0]); mask = np.array([True, True])
    pts = utils.generate_starting_points(nsMin, nsMax, central, mask=mask, n=2,
                                         start_method='nonsense',
                                         channels_and_bins=[('SR', 'SR', 2)], logger=log)
    assert pts.shape == (2, 2), pts.shape


# ---------------------------------------------------------------- bug 4
def test_start_method_key():
    from default_params import default_param_dict
    assert 'start_method' in default_param_dict, "'start_method' missing from defaults"
    assert 'start method' not in default_param_dict, "old 'start method' key still present"
    assert default_param_dict['start_method'] == 'default'


# ---------------------------------------------------------------- bug 5
def test_card_shapes():
    import yaml
    from collections import OrderedDict
    for f in sorted(os.listdir('cards')):
        if not f.endswith('.yaml') or 'kopia' in f:
            continue
        d = yaml.safe_load(open(os.path.join('cards', f)))
        ch = d.get('channels')
        assert isinstance(ch, list), f'{f}: channels must be a LIST of groups, got {type(ch).__name__}'
        assert isinstance(ch[0], dict), f'{f}: channel group 0 must be a mapping'
        # this is exactly what sample.py does
        OrderedDict(ch[0])


# ---------------------------------------------------------------- bug 6
def test_no_typo_keys():
    import yaml, utils
    docs = list(yaml.safe_load_all(open('parameters.yaml')))
    unknown = utils.check_unknown_parameters(docs, log)
    # 'analyses' lives in the defaults; nothing else should be unrecognised
    assert unknown == [], f'unrecognised parameter keys remain: {unknown}'


def test_unknown_key_is_detected():
    """The new guard must actually fire on a typo."""
    import utils
    found = utils.check_unknown_parameters([{'analysis': 'x', "SR_sigma'": 0.1, 'scan': 1}], log)
    assert found == ['SR_sigma\'', 'scan'], found


# ---------------------------------------------------------------- bug 7
def test_warning_bin_names():
    """Clipped-uncertainty and CR-spread warnings must name the right bins."""
    import utils
    records = []

    class Rec:
        def warning(self, m): records.append(m)
        def critical(self, m): records.append(m)
        def info(self, m): pass
        def debug(self, m): pass
        def error(self, m): records.append(m)

    cab = [('CR_a', 'CR', 2), ('SR_x', 'SR', 2)]
    names = ['CR_a-0', 'CR_a-1', 'SR_x-0', 'SR_x-1']
    B = [100.0, 100.0, 10.0, 10.0]
    # blow up the uncertainty on the LAST bin only
    dB = [5.0, 5.0, 1.0, 999.0]
    obs = [100.0, 130.0, 10.0, 10.0]
    bkg_yields = list(zip(names, B))
    bkg_unc = list(zip(names, dB))
    obs_yields = list(zip(names, obs))

    utils.get_scan_limits(bkg_yields, bkg_unc, obs_yields, cab,
                          True, False, 0.01, 0.10, 'both', 'both', 'obs', 'obs', Rec())
    clip = [m for m in records if 'clip it to 3' in m]
    assert clip, 'no clipping warning emitted'
    assert "'SR_x-1'" in clip[0], f'clipping warning names the wrong bin: {clip[0]}'
    assert 'SR_x-2' not in clip[0], f'clipping warning uses nbins as bin index: {clip[0]}'

    spread = [m for m in records if 'signal_leakage_CR_spread' in m]
    assert spread, 'no CR spread warning emitted'
    # the offending CR bin is CR_a-1 (obs 130 vs B 100, spread only 1%)
    assert 'CR_a-1' in spread[0], f'CR warning names the wrong bin: {spread[0]}'


# ---------------------------------------------------------------- rough edges
def test_get_obs_signal_returns():
    import utils
    out = utils.get_obs_signal([('a-0', 10.0), ('b-0', 5.0)], [('a-0', 12.0), ('b-0', 4.0)])
    assert out is not None, 'get_obs_signal still returns None'
    assert out == [('a-0', [2.0]), ('b-0', [-1.0])], out


def test_criteria_validation():
    src = open('likelihood.py').read()
    assert 'EXPLICIT_CRITERIA' in src and 'VALID_CRITERIA' in src
    assert "str(np.random.choice(['nLL_exp_mu1', 'nLL_obs_mu1']))" in src, \
        'ScanWrapper still passes a numpy array as the criterion'
    assert 'del nLL_exp_mu1, nLL_obs_mu1' not in src, 'dead del-after-return still present'


def test_check_for_nan_handles_inf():
    src = open('likelihood.py').read()
    i = src.index('def check_for_nan')
    body = src[i:i + 1200]
    assert 'isinf(likelihood)' in body, 'check_for_nan still ignores inf'


def test_wallclock_timing():
    src = open('sample.py').read()
    assert 'time.process_time_ns() - analysis_time' not in src, 'still timing with process_time'
    assert 'time.perf_counter_ns()' in src, 'perf_counter not used'


def test_no_variable_shadowing():
    src = open('sample.py').read()
    assert 'for ii in range(len(param_docs))' not in src, 'analysis loop still uses ii'
    assert 'channel_nbins = channel_item[1]' not in src, 'channel_nbins still shadowed'
    assert 'for analysis_index in range(len(param_docs))' in src
    assert 'n_bins_in_channel' in src


def test_defaults_not_mutated():
    src = open('sample.py').read()
    assert 'copy.deepcopy(default_param_dict)' in src, 'global_param_dict still aliases the defaults'


def test_fit_bkg_paths_defined():
    """input_bins_ordered & friends must be defined for BOTH fit_bkg branches."""
    src = open('sample.py').read()
    build = src.index('input_bins_ordered = []')
    branch = src.index("if not param_dict['fit_bkg']:")
    assert build < branch, 'input_bins_ordered is still built inside the not-fit_bkg branch only'
    fit_branch = src.index('cov_matrix = simplified_background_model')
    tail = src[fit_branch:fit_branch + 900]
    assert 'bkg_yields_ordered' in tail and 'bkg_unc_ordered' in tail, \
        'fit_bkg branch does not build the *_ordered lists'


def test_probe_mask_excludes_pinned_and_removed():
    """Pinned bins and bins of removed channels must not be probed."""
    import utils
    cab = [('CRa_cuts', 'CR', 3), ('VRb_cuts', 'VR', 2), ('SRc_cuts', 'SR', 2), ('SRd_cuts', 'SR', 1)]
    # signal leakage off in CRs and VRs -> those 5 bins are pinned
    scan_mask = utils.get_mask(8, cab, True, False, False)
    assert list(scan_mask) == [False]*5 + [True]*3, list(scan_mask)

    probe = utils.get_probe_mask(scan_mask, cab, ['SRd_cuts'])
    # 5 pinned + 1 removed SR bin -> only the 2 SRc bins get probed
    assert list(probe) == [False]*5 + [True, True, False], list(probe)

    # leakage on everywhere and nothing removed -> probe everything
    all_on = utils.get_probe_mask(utils.get_mask(8, cab, True, True, True), cab, [])
    assert all_on.all()
    # must not alias or mutate the mask it was given
    src = utils.get_mask(8, cab, True, True, True)
    utils.get_probe_mask(src, cab, ['CRa_cuts'])
    assert src.all(), 'get_probe_mask mutated the scan mask it was given'


def test_find_min_S_keeps_unprobed_limits():
    """Skipped bins must keep their incoming nSmin, not gain the +1e-4 margin."""
    import likelihood, inspect
    sig = inspect.signature(likelihood.find_min_S)
    assert 'probe_mask' in sig.parameters, 'find_min_S has no probe_mask parameter'

    src = inspect.getsource(likelihood.find_min_S)
    assert 'minimalS[~probe_mask] = nSmin[~probe_mask]' in src, \
        'unprobed bins do not inherit their incoming nSmin'
    assert 'if not probe_mask[ii+bb]:' in src, 'the probing loop does not skip unprobed bins'
    # the +1e-4 margin must SURVIVE for the bins that are actually probed
    assert 'bin_vals[bb] * mu+1e-4' in src, 'the +1e-4 safety margin was removed from probed bins'


def test_removal_resolved_before_find_min_S():
    """remove_channels must be resolved before find_min_S so it can skip them."""
    src = open('sample.py').read()
    removal = src.index('# Prepare channels for removal')
    probing = src.index('nSmin = find_min_S(')
    assert removal < probing, 'channel removal is still resolved after find_min_S'
    assert 'probe_mask=probe_mask' in src, 'find_min_S is not given the probe mask'


def test_merge_results_roundtrip():
    import utils
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k in range(3):
            p = os.path.join(d, f'table-{k}.csv')
            with open(p, 'w') as f:
                f.write('a,b\n')
                np.savetxt(f, np.array([[k, k + 1.0], [k + 2.0, k + 3.0]]), delimiter=',')
            paths.append(p)
        out, mn, mx = utils.merge_results(paths, keep_files=False, suffix='t', logger=log)
        assert os.path.isfile(out)
        rows = open(out).read().strip().split('\n')
        assert len(rows) == 7, f'expected header + 6 rows, got {len(rows)}'
        assert mn == [0.0, 1.0] and mx == [4.0, 5.0], (mn, mx)


if __name__ == '__main__':
    print('Verifying fixes...')
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            check(name[5:], fn)
    bad = [r for r in RESULTS if not r[0]]
    print(f'\n{len(RESULTS)-len(bad)}/{len(RESULTS)} checks passed')
    sys.exit(1 if bad else 0)

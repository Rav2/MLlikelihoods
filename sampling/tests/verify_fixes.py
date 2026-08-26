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


def _limits_fixture():
    """Two-channel model; card lists the channels in the REVERSE of model order."""
    import numpy as np
    model_bins = ['CRa_cuts-0', 'CRa_cuts-1', 'SRb_cuts-0']
    card_bins  = ['SRb_cuts-0', 'CRa_cuts-0', 'CRa_cuts-1']
    central = np.array([100.0, 50.0, 8.0])          # model order
    # card order: SRb(8), CRa-0(100), CRa-1(50)
    entry = [[2.0, 20.0], [90.0, 115.0], [45.0, 58.0]]
    return model_bins, card_bins, central, entry


def test_scan_limits_reorders_and_converts():
    """Limits are total yields in CARD order; must come back as offsets in MODEL order."""
    import utils, numpy as np
    model_bins, card_bins, central, entry = _limits_fixture()
    lo, hi = utils.load_scan_limits([entry], 0, 1, card_bins, model_bins, central, 'ps.json', log)
    # model order is CRa-0, CRa-1, SRb-0 -> totals (90,115), (45,58), (2,20)
    assert np.allclose(lo, [90.0-100.0, 45.0-50.0, 2.0-8.0]), lo
    assert np.allclose(hi, [115.0-100.0, 58.0-50.0, 20.0-8.0]), hi


def test_scan_limits_none_paths():
    import utils
    model_bins, card_bins, central, entry = _limits_fixture()
    assert utils.load_scan_limits(None, 0, 1, card_bins, model_bins, central, 'p', log) is None
    assert utils.load_scan_limits([None], 0, 1, card_bins, model_bins, central, 'p', log) is None
    # per-patchset selection
    out = utils.load_scan_limits([None, entry], 1, 2, card_bins, model_bins, central, 'p', log)
    assert out is not None


def test_scan_limits_validation():
    """Every malformed shape must raise, not silently sample the wrong box."""
    import utils
    model_bins, card_bins, central, entry = _limits_fixture()

    def expect_raise(sl, idx, n, why, bins=None):
        try:
            utils.load_scan_limits(sl, idx, n, bins or card_bins, model_bins, central, 'p', log)
        except ValueError:
            return
        raise AssertionError(f'no ValueError for {why}')

    expect_raise(entry, 0, 1, 'not nested per patchset')          # forgot the outer list
    expect_raise([entry], 0, 2, 'wrong number of patchsets')
    expect_raise([entry[:2]], 0, 1, 'wrong number of bins')
    expect_raise([[[1.0], [2.0, 3.0], [4.0, 5.0]]], 0, 1, 'pair of wrong length')
    expect_raise([[['x', 3.0], [2.0, 3.0], [4.0, 5.0]]], 0, 1, 'non-numeric value')
    expect_raise([[[20.0, 2.0], [90.0, 115.0], [45.0, 58.0]]], 0, 1, 'min above max')
    # central value outside its limits -> the classic "I stored signal offsets" mistake
    expect_raise([[[-6.0, 12.0], [-10.0, 15.0], [-5.0, 8.0]]], 0, 1, 'central outside limits')


def test_scan_limits_wired_into_sample():
    src = open('sample.py').read()
    assert 'load_scan_limits(' in src, 'sample.py never calls load_scan_limits'
    load = src.index('hardcoded_limits = load_scan_limits(')
    probe = src.index('nSmin = find_min_S(')
    assert load < probe, 'limits are loaded after find_min_S instead of instead of it'
    assert 'Scan limits mode: LOADED' in src and 'Scan limits mode: COMPUTED' in src, \
        'no log message distinguishing the two modes'
    # find_min_S must sit inside the else branch, i.e. be skipped when limits are loaded
    between = src[load:probe]
    assert 'else:' in between, 'find_min_S is not guarded by the loaded/computed branch'


def test_1911_12606_cards():
    """Both card variants must parse, agree on limits, and differ only in sig_rel_unc."""
    import yaml, json, os
    a = yaml.safe_load(open('cards/1911.12606.yaml'))
    b = yaml.safe_load(open('cards/1911.12606-sigunc20.yaml'))
    assert a['sig_rel_unc'] == 0.0, a['sig_rel_unc']
    assert b['sig_rel_unc'] == 0.20, b['sig_rel_unc']
    for c in (a, b):
        sl = c['scan_limits']
        assert len(sl) == len(c['patchsets']), 'scan_limits does not mirror patchsets'
        for i, entry in enumerate(sl):
            if entry is None:
                continue
            assert len(entry) == len(c['channels'][i]), \
                f'patchset {i}: {len(entry)} limit rows vs {len(c["channels"][i])} channels'
            assert all(isinstance(p, list) and len(p) == 2 and p[0] <= p[1] for p in entry)
    # measured: the limits come out identical at 0.0 and 0.20 on this workspace,
    # because the probe succeeds at mu=1 before the uncertainty can matter
    assert a['scan_limits'] == b['scan_limits'], 'the two variants disagree on scan limits'
    # each card's context must record its OWN uncertainty, and agree with the card
    for name, c in (('1911.12606', a), ('1911.12606-sigunc20', b)):
        assert abs(c['scan_limits_context']['sig_rel_unc'] - c['sig_rel_unc']) < 1e-12, \
            f'{name}: card sig_rel_unc {c["sig_rel_unc"]} != context {c["scan_limits_context"]["sig_rel_unc"]}'
        # whichever patchset is enabled must have limits stored for it
        for i, (ps, entry) in enumerate(zip(c['patchsets'], c['scan_limits'])):
            if ps[1] and entry is None:
                raise AssertionError(f'{name}: patchset {ps[0]} is enabled but has no scan_limits')
    assert a['sig_rel_unc'] == 0.0 and b['sig_rel_unc'] == 0.20
    # NOTE: which patchset each card enables is a user choice and deliberately
    # not asserted here - the cards are not required to be otherwise identical.


def test_signal_modifiers_shared():
    """The probe and the scan must build signal modifiers the same way."""
    import likelihood, numpy as np, inspect
    vals = np.array([-4.0, 2.0])

    plain = likelihood.build_signal_modifiers(vals, 0.0)
    assert [m['type'] for m in plain] == ['lumi', 'normfactor'], plain

    withunc = likelihood.build_signal_modifiers(vals, 0.20)
    assert withunc[0]['type'] == 'histosys', withunc[0]
    hi = list(withunc[0]['data']['hi_data'])
    lo = list(withunc[0]['data']['lo_data'])
    # the up-variation of a NEGATIVE signal is MORE negative than nominal -
    # this is exactly why the probe has to see it
    assert np.isclose(hi[0], -4.8), hi
    assert np.isclose(lo[0], 0.0), lo   # existing clamp: max(0, S*(1-unc))
    assert [m['type'] for m in withunc[1:]] == ['lumi', 'normfactor']

    # both call sites must go through this one builder
    scan_src = inspect.getsource(likelihood.LikelihoodCalculatorWrapper.inject_signal)
    probe_src = inspect.getsource(likelihood.find_min_S)
    assert 'build_signal_modifiers(' in scan_src, 'the scan does not use the shared builder'
    assert 'build_signal_modifiers(' in probe_src, 'the probe does not use the shared builder'
    assert 'sig_rel_unc' in inspect.signature(likelihood.find_min_S).parameters, \
        'find_min_S cannot see the signal uncertainty'


def test_probe_receives_signal_uncertainty():
    """find_min_S must actually pass the uncertainty into the injected patch."""
    import likelihood, numpy as np

    captured = []

    class FakeInterpreter:
        background_only_model = {}
        def inject_signal(self, channel, vals, modifiers=None):
            captured.append((channel, list(vals), modifiers))
        def make_patch(self):
            return {}

    class FakeModel:
        class backend:
            class manager:
                backend = None
        def likelihood(self, poi_test, expected):
            return 1.0          # always finite -> probe stops after one iteration

    orig = likelihood.WorkspaceInterpreter
    likelihood.WorkspaceInterpreter = lambda spec: FakeInterpreter()
    try:
        likelihood.find_min_S(1, {}, lambda **kw: FakeModel(), np.array([-5.0]),
                              [('SRx', 'SR', 1)], log, sig_rel_unc=0.20)
    finally:
        likelihood.WorkspaceInterpreter = orig

    assert captured, 'find_min_S never injected anything'
    _, _, mods = captured[0]
    assert mods is not None, 'probe injected with default modifiers, ignoring sig_rel_unc'
    assert mods[0]['type'] == 'histosys', f'no signal-uncertainty modifier in the probe: {mods}'


def _run_probe_with_likelihood(lik, nSmin, cab, niter, recorder):
    """Drive find_min_S against a stub model whose likelihood is ``lik(call_index)``."""
    import likelihood, numpy as np

    class FakeInterpreter:
        background_only_model = {}
        def inject_signal(self, channel, vals, modifiers=None):
            pass
        def make_patch(self):
            return {}

    state = {'n': 0}

    class FakeModel:
        class backend:
            class manager:
                backend = None
        def likelihood(self, poi_test, expected):
            # two calls (apriori, observed) per bisection step share one index
            v = lik(state['n'] // 2)
            state['n'] += 1
            return v

    orig = likelihood.WorkspaceInterpreter
    likelihood.WorkspaceInterpreter = lambda spec: FakeInterpreter()
    try:
        return likelihood.find_min_S(niter, {}, lambda **kw: FakeModel(),
                                     np.array(nSmin, dtype=float), cab, recorder)
    finally:
        likelihood.WorkspaceInterpreter = orig


class _Rec:
    """Minimal logger stand-in that keeps every message by level."""
    def __init__(self):
        self.msgs = []
    def info(self, m): self.msgs.append(('info', m))
    def warning(self, m): self.msgs.append(('warning', m))
    def error(self, m): self.msgs.append(('error', m))
    def critical(self, m): self.msgs.append(('critical', m))
    def errors(self): return [m for lvl, m in self.msgs if lvl == 'error']


def test_probe_reports_failure_at_mu1():
    """A non-finite likelihood at the candidate limit is a model/input problem: log ERROR."""
    rec = _Rec()
    # NaN on the first (mu=1) step, finite afterwards: the probe recovers at
    # mu=0.5 but the failure at the candidate limit must still be reported
    _run_probe_with_likelihood(lambda i: float('nan') if i == 0 else 1.0,
                               [-5.0], [('SRx', 'SR', 1)], 4, rec)
    errs = rec.errors()
    assert errs, 'no error logged when the likelihood was non-finite at mu=1'
    per_bin = [m for m in errs if 'SRx-0' in m and 'NON-FINITE LIKELIHOOD AT mu=1' in m]
    assert per_bin, f'no per-bin error naming the offending bin: {errs}'
    # the message has to be descriptive: what failed, and what to check
    assert 'nLL_exp_mu1' in per_bin[0], 'the message does not say which likelihood failed'
    assert 'bkg_yields' in per_bin[0], 'the message does not point at the likely cause'
    assert 'REDUCED' in per_bin[0], 'the message does not warn the limit is a fallback'
    summary = [m for m in errs if '1 of 1 probed bins' in m]
    assert summary, f'no end-of-probe summary of the failed bins: {errs}'


def test_probe_silent_when_model_is_sound():
    """No error when every bin evaluates at mu=1 - the check must not cry wolf."""
    rec = _Rec()
    _run_probe_with_likelihood(lambda i: 1.0, [-5.0, -3.0],
                               [('SRx', 'SR', 2)], 3, rec)
    assert not rec.errors(), f'errors logged for a healthy model: {rec.errors()}'


def test_probe_failure_below_mu1_is_not_an_error():
    """Bisection steps below mu=1 are expected to fail; only mu=1 is an error."""
    rec = _Rec()
    _run_probe_with_likelihood(lambda i: 1.0 if i == 0 else float('nan'),
                               [-5.0], [('SRx', 'SR', 1)], 4, rec)
    assert not rec.errors(), f'a sub-mu=1 failure was reported as an error: {rec.errors()}'


def test_region_pinning_applied_to_loaded_limits():
    """A loaded box is general; the run's own pinning must be re-imposed on it."""
    import utils, numpy as np
    cab = [('CRa_cuts', 'CR', 2), ('VRb_cuts', 'VR', 1), ('SRc_cuts', 'SR', 2)]
    lo = np.array([-30.0, -25.0, -8.0, -4.0, -3.0])
    hi = np.array([+30.0, +25.0, +8.0, +9.0, +7.0])

    # leakage on everywhere -> untouched
    a, b, n = utils.apply_region_pinning(lo, hi, cab, True, True, log)
    assert n == 0 and np.allclose(a, lo) and np.allclose(b, hi)

    # CR leakage off -> only the 2 CR bins collapse
    a, b, n = utils.apply_region_pinning(lo, hi, cab, False, True, log)
    assert n == 2, n
    assert np.allclose(a[:2], -1e-10) and np.allclose(b[:2], 1e-10)
    assert np.allclose(a[2:], lo[2:]), 'non-CR bins were altered'

    # both off -> CR and VR collapse, SR untouched
    a, b, n = utils.apply_region_pinning(lo, hi, cab, False, False, log)
    assert n == 3, n
    assert np.allclose(a[3:], lo[3:]) and np.allclose(b[3:], hi[3:])

    # must not mutate the caller's arrays
    assert np.allclose(lo, [-30.0, -25.0, -8.0, -4.0, -3.0]), 'input array was mutated'


def test_pinning_wired_after_load():
    src = open('sample.py').read()
    load = src.index('hardcoded_limits = load_scan_limits(')
    pin = src.index('apply_region_pinning(')
    probe = src.index('nSmin = find_min_S(')
    assert load < pin < probe, 'pinning is not applied to the loaded limits'


def test_harvest_prepares_general_config():
    """Harvest configs must probe every bin: leakage on, nothing removed."""
    import subprocess, tempfile, yaml, os, glob
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, 'tools/harvest_limits.py', 'prepare',
                            'cards/1912.08479.yaml', '1912.08479',
                            '--outdir', d, '--sig-rel-unc', '0.0'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cards = glob.glob(os.path.join(d, 'card-*.yaml'))
        assert cards, 'no harvest card generated'
        c = yaml.safe_load(open(cards[0]))
        assert c['signal_leakage_CR'] is True and c['signal_leakage_VR'] is True, \
            'harvest would pin regions and produce a non-general box'
        assert c['remove_channels'] == [] and c['removeCRsVRs'] is False, \
            'harvest would drop channels and produce a non-general box'
        assert 'scan_limits' not in c, 'harvest card still carries scan_limits'
        # exactly one patchset enabled
        assert sum(1 for p in c['patchsets'] if p[1]) == 1, c['patchsets']


def _ctx_params(sig=0.0, cr=0.5, vr=0.5, leak_cr=True, leak_vr=True):
    return {'sig_rel_unc': sig, 'signal_leakage_CR_spread': cr, 'signal_leakage_VR_spread': vr,
            'signal_leakage_CR': leak_cr, 'signal_leakage_VR': leak_vr}


HARVEST_CTX = {'sig_rel_unc': 0.0, 'signal_leakage_CR_spread': 0.5,
               'signal_leakage_VR_spread': 0.5, 'CR_center': 'obs', 'VR_center': 'obs'}


def test_context_guard_accepts_matching_run():
    import utils
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(), 'p', log) is True


def test_context_guard_rejects_changed_uncertainty():
    """ANY change to sig_rel_unc invalidates the limits, in either direction."""
    import utils
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(sig=0.20), 'p', log) is False
    ctx = dict(HARVEST_CTX, sig_rel_unc=0.20)
    assert utils.check_scan_limits_context(ctx, _ctx_params(sig=0.0), 'p', log) is False
    # and a match still passes
    assert utils.check_scan_limits_context(ctx, _ctx_params(sig=0.20), 'p', log) is True


def test_context_guard_rejects_wider_spread():
    """A spread WIDER than harvested pushes the scan past verified territory."""
    import utils
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(cr=0.6), 'p', log) is False
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(vr=0.9), 'p', log) is False
    # equal is fine; narrower is allowed (warns) because the box merely covers more
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(cr=0.5), 'p', log) is True
    assert utils.check_scan_limits_context(HARVEST_CTX, _ctx_params(cr=0.1), 'p', log) is True


def test_context_guard_ignores_pinned_regions():
    """A region with leakage off is pinned on load, so its spread is irrelevant."""
    import utils
    # CR spread far wider than harvested, but CR leakage is off -> still usable
    p = _ctx_params(cr=5.0, leak_cr=False)
    assert utils.check_scan_limits_context(HARVEST_CTX, p, 'p', log) is True
    # same spread but leakage ON -> rejected
    p = _ctx_params(cr=5.0, leak_cr=True)
    assert utils.check_scan_limits_context(HARVEST_CTX, p, 'p', log) is False


def test_context_guard_missing_context():
    """No context means nothing can be verified -> recompute, do not trust."""
    import utils
    assert utils.check_scan_limits_context(None, _ctx_params(), 'p', log) is False
    assert utils.check_scan_limits_context({}, _ctx_params(), 'p', log) is False
    try:
        utils.check_scan_limits_context(['not', 'a', 'mapping'], _ctx_params(), 'p', log)
    except ValueError:
        pass
    else:
        raise AssertionError('a non-mapping context should raise')


def test_context_guard_partial_context():
    """A setting this run depends on but the context omits is unverifiable."""
    import utils
    full = dict(HARVEST_CTX)

    # sig_rel_unc missing -> cannot verify the most important dimension
    no_sig = {k: v for k, v in full.items() if k != 'sig_rel_unc'}
    assert utils.check_scan_limits_context(no_sig, _ctx_params(), 'p', log) is False

    # CR spread missing while CR leakage is ON -> unverifiable
    no_cr = {k: v for k, v in full.items() if k != 'signal_leakage_CR_spread'}
    assert utils.check_scan_limits_context(no_cr, _ctx_params(leak_cr=True), 'p', log) is False

    # same context, but CR leakage OFF -> the CR spread is irrelevant, so it is fine
    assert utils.check_scan_limits_context(no_cr, _ctx_params(leak_cr=False), 'p', log) is True

    # VR spread missing with VR leakage off -> fine
    no_vr = {k: v for k, v in full.items() if k != 'signal_leakage_VR_spread'}
    assert utils.check_scan_limits_context(no_vr, _ctx_params(leak_vr=False), 'p', log) is True
    assert utils.check_scan_limits_context(no_vr, _ctx_params(leak_vr=True), 'p', log) is False


def test_loaded_message_reports_harvest_not_run():
    """The LOADED line must quote the HARVEST settings, never this run's."""
    src = open('sample.py').read()
    start = src.index('Scan limits mode: LOADED')
    block = src[max(0, start - 800):start + 400]
    assert 'harvest settings UNKNOWN' in block, \
        'no distinct wording when the harvest context is missing'
    assert "limits_ctx.get('sig_rel_unc'" in block, \
        'the LOADED line does not read sig_rel_unc from the harvest context'
    # the old bug: interpolating the run's own settings and calling them "harvested at"
    assert "harvested at \"\n" not in block
    assert "sig_rel_unc={param_dict['sig_rel_unc']}" not in block, \
        "the LOADED line still labels this run's sig_rel_unc as the harvested one"


def test_context_guard_wired_into_sample():
    src = open('sample.py').read()
    assert 'check_scan_limits_context(' in src, 'sample.py never checks the harvest context'
    guard = src.index('check_scan_limits_context(')
    probe = src.index('nSmin = find_min_S(')
    assert guard < probe, 'the guard runs after the probe'
    # a failed guard must fall back to computing
    tail = src[guard:guard + 400]
    assert 'hardcoded_limits = None' in tail, 'a failed guard does not fall back to recomputation'


def test_cards_carry_context():
    """Every card with scan_limits must record what it was harvested with."""
    import yaml, os
    for f in sorted(os.listdir('cards')):
        if not f.endswith('.yaml') or 'kopia' in f:
            continue
        c = yaml.safe_load(open(os.path.join('cards', f)))
        if not c.get('scan_limits'):
            continue
        ctx = c.get('scan_limits_context')
        assert isinstance(ctx, dict), f'{f}: scan_limits without scan_limits_context'
        for k in ('sig_rel_unc', 'signal_leakage_CR_spread', 'signal_leakage_VR_spread'):
            assert k in ctx, f'{f}: context missing {k}'
        if 'sig_rel_unc' in c:
            assert abs(c['sig_rel_unc'] - ctx['sig_rel_unc']) < 1e-12, \
                f'{f}: card sig_rel_unc {c["sig_rel_unc"]} != harvested {ctx["sig_rel_unc"]}'


def test_metadata_has_no_starting_points():
    """Starting points are reproducible from the seed; do not store them."""
    import utils, inspect
    sig = inspect.signature(utils.create_metadata)
    assert 'points' not in sig.parameters, 'create_metadata still takes a points argument'
    src = inspect.getsource(utils.create_metadata)
    assert "metadata['starting_points']" not in src, 'starting_points is still written'
    src_sample = open('sample.py').read()
    assert '[list(p + central_values) for p in p0s]' not in src_sample, \
        'sample.py still builds the starting-point list for the metadata'


def test_find_placeholder_rows():
    import utils, numpy as np
    from likelihood import NAN_PLACEHOLDER as P
    # 2 bin columns + 8 likelihood columns
    rows = np.array([
        [1.0, 2.0] + [3.0]*8,          # clean
        [1.0, 2.0] + [3.0]*7 + [P],    # placeholder in the last nLL
        [1.0, 2.0] + [-P] + [3.0]*7,   # negative placeholder (was -inf)
        [P,   2.0] + [3.0]*8,          # a huge YIELD is not a placeholder
    ])
    m = utils.find_placeholder_rows(rows)
    assert list(m) == [False, True, True, False], list(m)
    assert list(utils.find_placeholder_rows(np.empty((0, 10)))) == []


def test_merge_results_drops_placeholders():
    """The merged file must lose placeholder rows, and min/max must ignore them."""
    import utils, numpy as np, tempfile, os
    from likelihood import NAN_PLACEHOLDER as P

    msgs = []
    class Rec:
        def info(self, m): msgs.append(('info', m))
        def warning(self, m): msgs.append(('warn', m))
        def error(self, m): msgs.append(('error', m))
        def critical(self, m): msgs.append(('crit', m))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'table-0.csv')
        with open(p, 'w') as f:
            f.write('a,b,' + ','.join(f'nLL{i}' for i in range(8)) + '\n')
            np.savetxt(f, np.array([
                [1.0, 5.0] + [2.0]*8,
                [9.0, 9.0] + [2.0]*7 + [P],   # must be dropped
                [3.0, 7.0] + [4.0]*8,
            ]), delimiter=',')
        out, mn, mx = utils.merge_results([p], keep_files=True, logger=Rec())
        rows = [l for l in open(out).read().strip().split('\n')[1:] if l.strip()]
        assert len(rows) == 2, f'expected 2 surviving rows, got {len(rows)}'
        # the dropped row held the largest yields; min/max must not see them
        assert mx[0] == 3.0 and mx[1] == 7.0, (mx[0], mx[1])
        assert max(mx[-8:]) == 4.0, 'placeholder leaked into the recorded maximum'
        assert any('Dropped 1 of 3 rows' in m for _, m in msgs), msgs


def test_merge_results_all_rows_dropped():
    """If nothing survives, say so loudly and return empty min/max."""
    import utils, numpy as np, tempfile, os
    from likelihood import NAN_PLACEHOLDER as P

    msgs = []
    class Rec:
        def info(self, m): msgs.append(('info', m))
        def warning(self, m): msgs.append(('warn', m))
        def error(self, m): msgs.append(('error', m))
        def critical(self, m): msgs.append(('crit', m))

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'table-0.csv')
        with open(p, 'w') as f:
            f.write('a,b,' + ','.join(f'nLL{i}' for i in range(8)) + '\n')
            np.savetxt(f, np.array([[1.0, 5.0] + [P]*8,
                                    [3.0, 7.0] + [2.0]*7 + [P]]), delimiter=',')
        out, mn, mx = utils.merge_results([p], keep_files=True, logger=Rec())
        assert mn == [] and mx == [], (mn, mx)
        body = [l for l in open(out).read().strip().split('\n')[1:] if l.strip()]
        assert body == [], 'rows survived that should not have'
        crit = [m for lvl, m in msgs if lvl == 'crit']
        assert crit, 'no critical message when every row was dropped'
        assert 'NO USABLE ROWS LEFT' in crit[0]
        assert 'nLL7' in crit[0], 'the message does not name the likelihood columns'


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

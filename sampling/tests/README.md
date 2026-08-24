# Sampling regression tests

Small harness added alongside the bug-fix pass. Nothing here writes into
`../../tables/` - the parameter file points at a throwaway directory.

## Files

- `verify_fixes.py`  - unit checks, one per fixed bug. 16 of the 17 fail on
  the pre-fix code, so they are real regression guards rather than tautologies.
- `make_test_analysis.py` - builds a tiny synthetic pyhf workspace with
  MULTI-BIN channels (3+2+2+1). Multi-bin channels are what the channel-removal
  bookkeeping bug needed in order to show itself.
- `check_outputs.py` - validates a finished run: row counts, surviving bin
  columns, finiteness, scan limits, metadata completeness.

## Running

```bash
cd sampling
python tests/verify_fixes.py                       # unit checks (seconds)

python tests/make_test_analysis.py ../data/test-multibin
cp -r ../data/test-multibin ../data/test-fitbkg    # same workspace, fitted bkg
python sample.py parameters_test.yaml              # ~8 min, writes to /tmp
python tests/check_outputs.py /tmp/mll-sampling-test
```

`parameters_test.yaml` also samples the real `1912.08479`, so its archive has
to be present in `../data/`.

## The +1e-4 probe margin

`find_min_S()` probes with `bin_vals*mu + 1e-4` and keeps the probed value, so
a probed bin's lower limit sits 1e-4 above the value that was actually
verified finite. That margin is deliberate: it keeps the scan off the edge
where the likelihood breaks.

It must NOT reach bins the sampler never varies. A pinned bin (signal leakage
off for its region, so `calculate_sigmas` gives it step size 0) came in with
`nSmin = -1e-10`, and `-1e-10*mu + 1e-4` is *positive* - a floor above zero
under a bin that has to hold exactly zero signal. `NewStateWrapper` then
clipped it up, and the column came out mixing `central` and `central + 1e-4`
instead of being constant. Bins of removed channels had the same margin
computed and then thrown away.

`find_min_S()` now takes a `probe_mask` (built by `utils.get_probe_mask`) and
skips both kinds: they keep their incoming `nSmin` and cost no model
evaluations. `check_outputs.py` asserts pinned bins are exactly constant.

Related, still open: `find_min_S` dominates wall time on large workspaces
(2h10m of a 2h35m production run on 1911.12606), because each probed bin
rebuilds the full spey model and runs two fits, serially. And on a bin whose
bisection iterates, `minimalS` keeps the *last* successful probe rather than
the most negative one, so the result depends on where the loop stops. Neither
is a correctness problem for the workspaces in `cards/`, where `mu=1` succeeds
on the first try for every bin.

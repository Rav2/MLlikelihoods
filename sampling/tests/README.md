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

## Known wart (pre-existing, not fixed)

`find_min_S()` probes with `bin_vals*mu + 1e-4` and keeps that value, so every
recorded lower limit sits exactly 1e-4 above the true one. For non-scanned
CR/VR bins that turns a limit of ~0 into a positive floor of +1e-4, and the
MCMC clip nudges those bins to `central + 1e-4`. Physically negligible at
yields of O(10-100), but it is a systematic offset. `check_outputs.py`
tolerates it explicitly rather than silently.

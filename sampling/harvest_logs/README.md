# Harvest logs

Logs of the runs whose `Scan limits` tables were hardcoded into the analysis
cards as `scan_limits`. Kept so the stored numbers can be traced back to the
run that produced them; each card also records the metadata file and seed it
took its values from.

| log | analysis | patchset | sig_rel_unc | bins |
|---|---|---|---|---|
| `1911_12606-ps0-sig0_0.txt` | 1911.12606 | EWKinos | 0.0 | 50 |
| `1911_12606-ps0-sig0_2.txt` | 1911.12606 | EWKinos | 0.20 | 50 |
| `1911_12606-ps2-sig0_0.txt` | 1911.12606 | Sleptons | 0.0 | 38 |
| `1911_12606-ps2-sig0_2.txt` | 1911.12606 | Sleptons | 0.20 | 38 |
| `1911_06660-sig0_0.txt` | 1911.06660 | combined | 0.0 | 5 |
| `1912_08479-sig0_0.txt` | 1912.08479 | patchset | 0.0 | 4 |

All harvested with `low_lim_samples = 50`, every bin probed (leakage enabled in
every region, nothing removed), so the stored box is general. 1911.12606 used
`signal_leakage_CR_spread = 0.5` to match its production configuration; the two
small analyses used the default 0.10.

## Take the numbers from the metadata, not from these tables

The `Scan limits` table is printed with `np.round(..., 1)`. On the earlier
1911.12606 runs that rounding moved limits by up to 0.05 events and pushed the
upper limit *below* its own central value in 34 of 88 bins. The cards therefore
carry the full-precision `lower_limits` / `upper_limits` from each run's
metadata JSON; `tools/harvest_limits.py merge` reads those directly.

## Signal uncertainty

The probe now injects the same `histosys` signal-uncertainty modifier the scan
uses, so limits *can* depend on `sig_rel_unc`. Measured on these workspaces
they do not: EWKinos (50 bins) and Sleptons (38 bins) give bit-identical limits
at 0.0 and 0.20, because the probe succeeds at `mu = 1` on its first iteration
for every bin and so never reaches the region where the likelihood breaks. The
two 1911.12606 cards consequently carry the same numbers - harvested
separately, and verified equal rather than assumed.

## Not harvested

- **2102.10874** - no workspace in `data/` at all, and its card points at a
  placeholder `dummy.json`.
- **1909.09226** - its card supplies 7 `bkg_yields` but the workspace has 14
  bins (`SRHMEM_mct2`, `SRMMEM_mct2`, `SRLMEM_mct2` are 3-bin channels), so it
  cannot run until the missing values are filled in.
- **1908.08215** and **2106.01676** - too slow to probe here; use
  `cluster_scripts/harvest_limits_1908_08215.sh` and
  `cluster_scripts/harvest_limits_2106_01676.sh`, which run the probe and write
  the limits into the cards automatically.

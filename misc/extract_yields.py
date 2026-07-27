#!/usr/bin/env python3
"""
extract_yields.py
-----------------
Extract total expected yields (background + signal at given mu) for every signal
patch in an ATLAS HistFactory / pyhf patchset, and write results to CSV.

Usage
-----
    python extract_yields.py bkg_onshell.json onshell_winobino_plus_patchset.json \
        [-o yields.csv]

Arguments
---------
    background  : Path to the background-only workspace JSON (pyhf schema).
    patchset    : Path to the pyhf PatchSet JSON (signal patches).
    -o / --output : Output CSV path (default: yields.csv).

Output
------
    CSV with one row per signal patch.
    Identifier columns:  patch_name, mC1_GeV, mN1_GeV
    Yield columns:       <channel_name>-<bin_index>   (e.g. SR1_WZ_cuts-0)
    All yields are rounded to 5 decimal places.

Notes
-----
    - Only actualdata yields are collected (auxdata/constraint terms excluded).
    - Channel order is preserved exactly as returned by pyhf.
    - Parameters are set to their suggested (nominal) init values, with the
      POI (mu_SIG) fixed to 1.0.  All nuisance parameters remain at their
      nominal values (i.e. no pre-fit to the patched model is performed;
      this gives the raw predicted yields at signal strength = 1).
"""

import argparse
import csv
import json
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    """Load and return a JSON file."""
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        sys.exit(f"ERROR: File not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: Could not parse JSON from {path}: {exc}")


def build_bin_names(spec: dict, model) -> list[str]:
    """
    Return bin labels in the channel order defined by the workspace spec JSON.

    pyhf internally sorts model.config.channels alphabetically, which does NOT
    match the original channel order from the JSON spec.  We therefore derive
    the order directly from spec['channels'] (insertion order = JSON order) so
    that column headers in the CSV match the ordering the user expects.

    Labels: '<channel_name>-<bin_index>' with bin_index starting at 0.
    """
    names = []
    for ch in spec["channels"]:
        ch_name = ch["name"]
        n_bins = model.config.channel_nbins[ch_name]
        for b in range(n_bins):
            names.append(f"{ch_name}-{b}")
    return names


def spec_channel_order(spec: dict) -> list[str]:
    """Return channel names in the order they appear in the spec JSON."""
    return [ch["name"] for ch in spec["channels"]]


def expected_yields_at_mu(spec: dict, model, mu=1.0) -> list[float]:
    """
    Return expected actualdata yields with all nuisances at nominal and mu=1,
    ordered to match the channel sequence in the workspace spec JSON.

    pyhf's model.expected_actualdata() returns bins in model.config.channels
    order, which is alphabetically sorted.  We build a permutation index to
    reorder those yields into spec JSON order before returning.
    """
    pars = model.config.suggested_init()
    pars[model.config.poi_index] = mu

    # Yields in pyhf's internal (alphabetically sorted) channel order.
    raw = model.expected_actualdata(pars)

    # --- Build the permutation: sorted_order → spec_order -------------------
    # Flat list of bin labels in pyhf's sorted order (matches raw exactly).
    sorted_bin_labels = [
        f"{ch_name}-{b}"
        for ch_name in model.config.channels
        for b in range(model.config.channel_nbins[ch_name])
    ]
    # Flat list of bin labels in the desired spec JSON order.
    spec_bin_labels = [
        f"{ch['name']}-{b}"
        for ch in spec["channels"]
        for b in range(model.config.channel_nbins[ch["name"]])
    ]
    # Index into sorted_bin_labels for each position in spec_bin_labels.
    sorted_index = {label: i for i, label in enumerate(sorted_bin_labels)}
    perm = [sorted_index[label] for label in spec_bin_labels]

    return [round(float(raw[i]), 5) for i in perm]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract total expected yields (bkg+sig at mu=1) for "
                    "every signal patch and write to CSV.",
    )
    parser.add_argument(
        "background",
        metavar="BACKGROUND_JSON",
        help="Path to the background-only pyhf workspace JSON.",
    )
    parser.add_argument(
        "patchset",
        metavar="PATCHSET_JSON",
        help="Path to the pyhf PatchSet JSON containing signal patches.",
    )
    parser.add_argument(
        "-o", "--output",
        default="yields.csv",
        metavar="OUTPUT_CSV",
        help="Output CSV file path (default: yields.csv).",
    )
    parser.add_argument(
        "-m", "--mu",
        default=1.0,
        metavar="MU",
        help="POI value.",
    )
    args = parser.parse_args()

    # --- Import pyhf here so the argparse --help works even without it ------
    try:
        import pyhf
    except ImportError:
        sys.exit(
            "ERROR: pyhf is not installed.\n"
            "Install it with:  pip install pyhf"
        )

    # --- Load inputs ---------------------------------------------------------
    print(f"Loading background workspace: {args.background}")
    bkg_spec = load_json(args.background)

    print(f"Loading signal patchset:      {args.patchset}")
    patchset = pyhf.PatchSet(load_json(args.patchset))

    n_patches = len(patchset.patches)
    print(f"Found {n_patches} signal patches.\n")

    # --- Determine bin names from the first patched model -------------------
    # All patches operate on the same channel layout, so we only need to do
    # this once.  We build it from the very first patch.
    first_patched_spec = patchset.apply(bkg_spec, patchset.patches[0].name)
    first_ws    = pyhf.Workspace(first_patched_spec)
    first_model = first_ws.model()
    bin_names   = build_bin_names(first_patched_spec, first_model)

    print(f"Channels / bins per model: {len(first_model.config.channels)} channels, "
          f"{len(bin_names)} bins total")
    print("Bin names:", bin_names)
    print()

    # --- Iterate patches and collect yields ----------------------------------
    fieldnames = ["patch_name", "mC1_GeV", "mN1_GeV"] + bin_names
    rows = []

    for i, patch in enumerate(patchset.patches):
        name   = patch.name
        values = patch.metadata.get("values", [None, None])
        mC1, mN1 = values[0], values[1]

        # Apply the patch and build the full (bkg + sig) model
        patched_spec = patchset.apply(bkg_spec, name)
        ws    = pyhf.Workspace(patched_spec)
        model = ws.model()

        # Yields at nominal nuisances, mu_SIG = 1
        yields = expected_yields_at_mu(patched_spec, model, args.mu)

        if len(yields) != len(bin_names):
            sys.exit(
                f"ERROR: patch '{name}' produced {len(yields)} yield values "
                f"but expected {len(bin_names)}.  Channel layout may differ "
                f"across patches, which is not supported."
            )

        row = {"patch_name": name, "mC1_GeV": mC1, "mN1_GeV": mN1}
        row.update(dict(zip(bin_names, yields)))
        rows.append(row)

        print(f"  [{i+1:3d}/{n_patches}]  {name:<45s}  mC1={mC1}, mN1={mN1}")

    # --- Write CSV -----------------------------------------------------------
    print(f"\nWriting {len(rows)} rows to: {args.output}")
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
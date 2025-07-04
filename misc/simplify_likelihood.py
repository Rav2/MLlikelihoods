import pyhf
import json

import simplify

bkg_path = '../data/1911.12606/EWKinos_bkgonly.json'
patch_path = '../data/1911.12606/EWKinos_patchset.json'
# set the computational backend to pyhf and load LH
pyhf.set_backend(pyhf.tensorlib, "minuit")
spec = json.load(open(bkg_path, "r"))
patchset = pyhf.PatchSet(json.load(open(patch_path)))
# ws from full LH
patch = patchset.patches[0]
ws = pyhf.Workspace(patch.apply(spec))

# get model and data for each ws we just created
# use polynomial interpolation and exponential extrapolation
# for nuisance params
model = ws.model(
    modifier_settings = {
        "normsys": {"interpcode": "code4"},
        "histosys": {"interpcode": "code4p"},
    }
)
data = ws.data(model)

# run fit
fit_result = simplify.fitter.fit(ws)

# plot the pulls
plt = simplify.plot.pulls(
    fit_result,
    "plots/"
)

# plot correlation matrix
plt = simplify.plot.correlation_matrix(
    fit_result,
    "plots/",
    pruning_threshold=0.1
)

# get a yieldstable in nice LaTeX format
tables = simplify.plot.yieldsTable(
    ws,
    "plots/",
    fit_result,
)

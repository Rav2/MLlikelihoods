"""Build a tiny synthetic analysis for exercising sample.py.

Deliberately uses MULTI-BIN channels, which is what the channel-removal
bookkeeping bug needed in order to show itself, plus a CR / VR / two SRs so
every branch of the region handling is touched.
"""
import json, os, sys
import pyhf

OUT = sys.argv[1] if len(sys.argv) > 1 else '../data/test-multibin'
os.makedirs(OUT, exist_ok=True)

# channel name -> (background yields per bin, observed yields per bin)
LAYOUT = [
    ('CRmulti_cuts', [120.0, 90.0, 60.0], [126.0, 88.0, 64.0]),   # 3 bins
    ('VRpair_cuts',  [40.0, 25.0],        [42.0, 23.0]),          # 2 bins
    ('SRlow_cuts',   [8.0, 5.0],          [10.0, 4.0]),           # 2 bins
    ('SRhigh_cuts',  [3.0],               [2.0]),                 # 1 bin
]

channels, observations = [], []
for name, bkg, obs in LAYOUT:
    channels.append({
        'name': name,
        'samples': [{
            'name': 'bkg',
            'data': bkg,
            'modifiers': [
                {'name': 'lumi', 'type': 'lumi', 'data': None},
                {'name': f'staterr_{name}', 'type': 'staterror',
                 'data': [round(0.10 * b, 4) for b in bkg]},
                {'name': 'theory_unc', 'type': 'normsys',
                 'data': {'hi': 1.06, 'lo': 0.94}},
            ],
        }],
    })
    observations.append({'name': name, 'data': obs})

workspace = {
    'channels': channels,
    'observations': observations,
    'measurements': [{
        'name': 'Test',
        'config': {
            'poi': 'mu_SIG',
            'parameters': [{
                'name': 'lumi', 'auxdata': [1.0], 'bounds': [[0.9, 1.1]],
                'inits': [1.0], 'sigmas': [0.017],
            }],
        },
    }],
    'version': '1.0.0',
}

with open(os.path.join(OUT, 'bkgonly.json'), 'w') as f:
    json.dump(workspace, f, indent=1)

# A patchset with one patch: adds a signal sample to every channel.
patch_ops = []
for idx, (name, bkg, obs) in enumerate(LAYOUT):
    sig = [round(0.25 * b, 4) for b in bkg]
    patch_ops.append({
        'op': 'add',
        'path': f'/channels/{idx}/samples/0',
        'value': {
            'name': 'signal',
            'data': sig,
            'modifiers': [
                {'name': 'lumi', 'type': 'lumi', 'data': None},
                {'name': 'mu_SIG', 'type': 'normfactor', 'data': None},
            ],
        },
    })

patchset = {
    'metadata': {
        'references': {'hepdata': 'ins0000000'},
        'digests': {'md5': '0' * 32},
        'labels': ['mass'],
        'description': 'synthetic patchset for testing the sampler',
    },
    'patches': [{'metadata': {'name': 'point_400', 'values': [400]}, 'patch': patch_ops}],
    'version': '1.0.0',
}

with open(os.path.join(OUT, 'patchset.json'), 'w') as f:
    json.dump(patchset, f, indent=1)

# validate against pyhf's own schemas before handing it to the sampler
ws = pyhf.Workspace(workspace)
ps = pyhf.PatchSet(patchset)
model = ws.model(patches=[ps.patches[0].patch], measurement_name='Test')
print('workspace OK  ->', OUT)
print('  channel_nbins :', dict(model.config.channel_nbins))
print('  total bins    :', sum(model.config.channel_nbins.values()))
print('  patches       :', len(ps.patches))
print('  poi           :', model.config.poi_name)

# emit the yields so they can be pasted into the analysis card
for name, bkg, obs in LAYOUT:
    print(f'  {name}: bkg={bkg} obs={obs}')

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import griddata

# Load your primary data for scatter plot
m1, m2, _,_,cls_exp, cls_obs = np.loadtxt('limit.txt').T

# Load exclusion lines from JSON
with open('./exclusion_lines.json') as f:
    exclusion_data = json.load(f)

# Extract observed and expected exclusion lines
obs_x = exclusion_data["TStauStau"]["obsExclusion_[[x,y],[x,y]]"]["x"]
obs_y = exclusion_data["TStauStau"]["obsExclusion_[[x,y],[x,y]]"]["y"]
exp_x = exclusion_data["TStauStau"]["expExclusion_[[x,y],[x,y]]"]["x"]
exp_y = exclusion_data["TStauStau"]["expExclusion_[[x,y],[x,y]]"]["y"]

# Set up a grid for smoothing
grid_x, grid_y = np.mgrid[min(m1):max(m1):25j, min(m2):max(m2):25j]

# Interpolate both cls_obs and cls_exp on the grid
cls_obs_grid = griddata((m1, m2), cls_obs, (grid_x, grid_y), method='linear')
cls_exp_grid = griddata((m1, m2), cls_exp, (grid_x, grid_y), method='linear')

# Plot smoothed contours
plt.figure(figsize=(8, 6))
plt.scatter(grid_x, grid_y, c=cls_exp_grid, cmap='tab20c')
plt.colorbar()
plt.contour(grid_x, grid_y, cls_obs_grid, colors='b', levels=[0.0, 0.95, 1.01], linestyles='-',)
plt.contour(grid_x, grid_y, cls_exp_grid, colors='b', levels=[0.0, 0.95, 1.01], linestyles=':',)

# Plot observed and expected exclusion lines
plt.plot(obs_x, obs_y, 'k-', label='Observed (Official)')
plt.plot(exp_x, exp_y, 'k:', label='Expected (Official)')

# Label axes and title
plt.xlabel('m1')
plt.ylabel('m2')
plt.title('Smooth Contours in m1-m2 plane with exclusion lines')
plt.legend()

# Show plot
plt.show()

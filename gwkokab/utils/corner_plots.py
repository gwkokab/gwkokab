import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# Generate two sets of data
data1_load = h5py.File('myout.hdf5','r')
data2_load = h5py.File('myout_org.hdf5','r')
data_3D1 = np.array(data1_load['pos'])
data_3D2 = np.array(data2_load['pos'])
org_steps = data_3D1.shape[0]
burn_steps = 2500

#Burning the initial walkers, you can choose the burning steps by looking at chain plots
data1_burn = data_3D1[burn_steps:]
data2_burn = data_3D2[burn_steps:]
walker = data1_burn.shape[1] 
steps = data1_burn.shape[0]
points = walker * steps
data1 = data1_burn.reshape((points,5))
data2 = data2_burn.reshape((points,4))
print(data2.shape)
print(data1.shape)
mean1 = [2,-1,10,50,0.05]
mean2 = [2,-1,10,50]
# Create DataFrames and add an empty fifth column to df2
df1 = pd.DataFrame(data1, columns=[r"$log_{10}(\frac{\mathcal{R}}{Gpc^{-3}yr^-1})$", r"$\alpha$", r"$m_{min} [M_\odot$]",r"$m_{max} [M_\odot]$",
        r"$\sigma_\epsilon$"])
df1['Dataset'] = 'EBBH'

df2 = pd.DataFrame(data2, columns=[r"$log_{10}(\frac{\mathcal{R}}{Gpc^{-3}yr^-1})$", r"$\alpha$", r"$m_{min} [M_\odot$]",r"$m_{max} [M_\odot]$"])
df2['$\sigma_\epsilon$'] = np.nan  # Add an empty fifth column
df2['Dataset'] = 'CBBH'

# Combine the datasets
combined_df = pd.concat([df1, df2])

# Define your custom axis ranges here
axis_ranges = {
    'rate': (1.7, 2.3),
    'alpha': (-2.2, 0.4),
    'm_min': (4, 13),
    'm_max': (47.5, 55),
    'sigma_e': (0.035, 0.065)
}

custom_colors = {
    'EBBH': 'orange',
    'CBBH': 'black'
}


# Initialize the PairGrid with your data
g = sns.PairGrid(combined_df, hue="Dataset", palette=custom_colors, diag_sharey=False)
g.map_lower(sns.kdeplot, alpha=0.5,levels=5, thresh =0.2, fill=False)
g.map_diag(sns.histplot, kde=False)

# Remove the upper right axes
for i in range(len(g.axes)):
    for j in range(i+1, len(g.axes)):
        g.axes[i, j].set_visible(False)

# Apply custom axis ranges
for i, row in enumerate(g.axes):
    for j, ax in enumerate(row):
        if i < len(axis_ranges) and j < len(axis_ranges):  # Check to avoid index errors
            param = list(axis_ranges.keys())[j]  # Get the parameter name for the column
            ax.set_xlim(axis_ranges[param])  # Set x-axis range for all plots
            if i != j:  # For off-diagonal plots only
                param = list(axis_ranges.keys())[i]  # Get the parameter name for the row
                ax.set_ylim(axis_ranges[param])  # Set y-axis range

# Add true value lines after setting the axis ranges
for i in range(len(g.axes)):
    for j in range(len(g.axes)):
        if i == j:  # Diagonal: Add vertical lines for true values
            g.axes[i, j].axvline(x=mean1[i], color='red', linestyle='--')
        elif i > j:  # Lower triangle: Add both horizontal and vertical lines for true values
            g.axes[i, j].axvline(x=mean1[j], color='red', linestyle='--')
            g.axes[i, j].axhline(y=mean1[i], color='red', linestyle='--')
            # Place a marker where the true value lines intersect
            g.axes[i, j].scatter(mean1[j], mean1[i], color='red', s=40, zorder=5)


#g.add_legend(loc='upper right', fontsize=14)
#egend = g._legend
#legend.set_title(None)
plt.savefig('new_corner_015.png')
plt.show()

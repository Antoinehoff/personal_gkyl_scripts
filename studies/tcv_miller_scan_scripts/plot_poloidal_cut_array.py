import os
import matplotlib.pyplot as plt
from scan_analysis import ScanMetadata
h5file = 'tcv_miller_scan_big_metadata_frame_500_navg_25.h5'
miller_scan = ScanMetadata(h5file)

# Generate the table of poloidal cuts for the paper
poloidal_cut_dir = '/Users/ahoffman/Desktop/simulation_gallery/tcv_miller_scan_5d_figures/poloidal_plots/cropped_images/'

field = 'Ti'

height_ratios = [ k_ for k_ in miller_scan.scan_params['kappa'][::-1]]
# Create a figure 
fig_aspect_ratio = 11/9
fig_size = 8.0
fig, axes = plt.subplots(ncols=len(miller_scan.scan_params['delta']), 
                         nrows=len(miller_scan.scan_params['kappa']),
                         height_ratios=height_ratios,
                         figsize=(fig_size, fig_aspect_ratio*fig_size))
# Minimize the space between subplots (tight_layout would override these)
fig.subplots_adjust(left=0.03, right=1.0, top=1.0, bottom=0.03*fig_aspect_ratio, wspace=-0.5, hspace=0.05)

write_delta_eq = True
write_kappa_eq = True
nkappa = len(miller_scan.scan_params['kappa'])
for delta in miller_scan.scan_params['delta'][:]:
    for kappa in miller_scan.scan_params['kappa'][:]:
        filename = f'{field}_delta_{delta:1.2f}_kappa_{kappa:1.2f}_power_5.0e+05_frame_500.png'
        #check if file exists
        if not os.path.exists(poloidal_cut_dir + filename):
            print(f'File {filename} does not exists.')
            
        # Load the png
        print(f'Loading {filename}...')
        img = plt.imread(poloidal_cut_dir + filename)
        # Plot the image in the correct subplot
        ax = axes[nkappa-miller_scan.scan_params['kappa'].index(kappa)-1, miller_scan.scan_params['delta'].index(delta)]
        ax.imshow(img, aspect='equal')
        # Remove axis ticks and lines
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        # If this is the first column, add a y-label
        if miller_scan.scan_params['delta'].index(delta) == 0:
            if write_kappa_eq:
                ax.set_ylabel(r'$\kappa=$'+f'{kappa:1.2f}', fontsize=12)
                # write_kappa_eq = False
            else:
                ax.set_ylabel(f'{kappa:1.2f}', fontsize=12)
        # If this is the last row, add an x-label
        if miller_scan.scan_params['kappa'].index(kappa) == 0:
            if write_delta_eq:
                ax.set_xlabel(r'$\delta=$'+f'{delta:1.2f}', fontsize=12)
                # write_delta_eq = False
            else:
                ax.set_xlabel(f'{delta:1.2f}', fontsize=12)

plt.savefig(f'/Users/ahoffman/writing/gkyl_tcv_miller_scan/figures/{field}_poloidal_cuts_grid.png', dpi=600)
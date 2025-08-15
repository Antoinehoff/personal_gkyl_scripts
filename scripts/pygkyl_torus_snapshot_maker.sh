#!bin/bash

simdir='/pscratch/sd/t/tnbernar/ceda-data/d3d/negD-3x/fix-q-adpt-src/'
fileprefix='gk_d3d_iwl_adapt_source_3x2v_p1'

label='5D simulation of DIII-D\nnegative triangularity edge plasma\n(PPPL & General Atomics)'

logo_path='/pscratch/sd/a/ah1032/Hollywood/gkeyll_logo.png'

movie_type='gif'
cameras='global'
config='d3d_nt'
rholim='2 60' # marche pas

plottype='snapshot'

fieldname='ni'
clim='0 3e19'

cameras='zoom_obmp'
img_size='800 600'


python3 pygkyl_plottorus.py --sim_dir=$simdir \
    --file_prefix=$fileprefix \
    --plot_type=$plottype \
    --field_name $fieldname \
    --clim $clim \
    --movie_type $movie_type \
    --cameras $cameras \
    --device_config $config \
    --cameras $cameras \
    --img_size $img_size \
    --logo_path $logo_path \
    --additional_text "$label" \

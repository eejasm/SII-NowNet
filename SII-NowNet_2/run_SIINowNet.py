import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from keras import Model, Input
from keras.layers import Layer

import numpy as np
from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from satpy import Scene
from pyproj import Transformer

from glob import glob
import os
import sys
import geopandas as gpd


# special class made for allowing the models (developed in keras 2) to be loaded in keras 3
class TFSMLayer(Layer):
    def __init__(self, filepath, call_endpoint="serving_default", **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.call_endpoint = call_endpoint
        self._infer = None  # will be loaded in build()

    def build(self, input_shape):
        # Load SavedModel once; save the TF function to self._infer
        self._loaded_model = tf.saved_model.load(self.filepath)
        self._infer = self._loaded_model.signatures[self.call_endpoint]
        super().build(input_shape)

    def call(self, inputs):
        # Ensure inputs are a tensor, pass through TF function, return the tensor output
        outputs = self._infer(inputs)
        
        # Pick the actual tensor output key here (adjust if needed)
        #return outputs['max_pooling2d_4']
        return list(outputs.values())[0] 

    def compute_output_shape(self, input_shape):
        # Return the expected output shape, for example (None, 26, 26, 1)
        # You may need to adapt this based on your actual model output
        batch_size = input_shape[0]
        return (batch_size, 26, 26, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filepath': self.filepath,
            'call_endpoint': self.call_endpoint,
        })
        return config


def save_initiation_fig(nowcast_data, initialisation_time, lead_time, file_path):
    
    fig = plt.figure(figsize=(4,4))
    
    msg_projection = ccrs.Geostationary(central_longitude=0.0,satellite_height=35785831)
    
    x_coords = np.load('sa_x_values.npy')
    y_coords = np.load('sa_y_values.npy')
    x_min, x_max, y_min, y_max = x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()

    # INITIATION
    tol_precip_colors = ["gainsboro","gainsboro","lightblue", "#6195CF","#F7CB45","#EE8026","#DC050C","#A5170E","#72190E","#882E72","#000000"]
    precip_colormap = mpl.colors.ListedColormap(tol_precip_colors[:])
    bounds = [0,0.01,0.02,0.03,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    norm = mpl.colors.BoundaryNorm(bounds,12, extend='both')
    
    ax = fig.add_subplot(111,projection=msg_projection)
    ax.coastlines()
    ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max], transform=msg_projection)
    im= ax.contourf(nowcast_data, origin = 'lower', transform=msg_projection,extent = [x_min,x_max,y_min,y_max],cmap=precip_colormap)
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.right_labels = gl.top_labels = False
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.set_title(initialisation_time[6:8]+'/'+initialisation_time[4:6]+'/'+initialisation_time[:4] + ' initiation (+'+lead_time+' hour)\n' + 'Initialisation time (UTC): ' + initialisation_time[8:10] + ':' + initialisation_time[10:12])

    province_gdf = gpd.read_file('/home/users/eejasm/SAWS/provinces/Province_New.shp')
    # Remove missing or empty geometries
    province_gdf = province_gdf[province_gdf.geometry.notnull()]
    province_gdf = province_gdf[~province_gdf.geometry.is_empty]

    # Ensure lon/lat CRS
    if province_gdf.crs != "EPSG:4326":
        province_gdf = province_gdf.to_crs(epsg=4326)

    # Plot provinces
    for geom in province_gdf.geometry:
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black')
    
    divider = make_axes_locatable(ax)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=precip_colormap,norm=norm), label='Probability',extend='neither',ticks=bounds, cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0, axes_class=maxes.Axes))
    
    date_str = initialisation_time[:8]  # YYYYMMDD
    output_dir = os.path.join(nowcast_fig_file_path, date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_dir+'/'+initialisation_time+'_initiation_'+lead_time+'.png', bbox_inches='tight')


def save_intensification_fig(nowcast_data, initialisation_time, lead_time, file_path):
    
    fig = plt.figure(figsize=(4,4))
    
    msg_projection = ccrs.Geostationary(central_longitude=0.0,satellite_height=35785831)
    
    x_coords = np.load('sa_x_values.npy')
    y_coords = np.load('sa_y_values.npy')
    x_min, x_max, y_min, y_max = x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()
    
    #INTENSIFICATION
    tol_precip_colors = ["gainsboro", "lightblue", "#6195CF","#F7CB45","#EE8026","#DC050C","#A5170E","#72190E","#882E72","#000000"]
    precip_colormap = mpl.colors.ListedColormap(tol_precip_colors[:])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    ax = fig.add_subplot(111,projection=msg_projection)
    ax.coastlines()
    ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max], transform=msg_projection)
    im= ax.contourf(nowcast_data, origin = 'lower', transform=msg_projection, extent = [x_min,x_max,y_min,y_max],cmap=precip_colormap)
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.right_labels = gl.top_labels = False
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.set_title(initialisation_time[6:8]+'/'+initialisation_time[4:6]+'/'+initialisation_time[:4] + ' intensification (+'+lead_time+' hour)\n' + 'Initialisation time (UTC): ' + initialisation_time[8:10] + ':' + initialisation_time[10:12])

    province_gdf = gpd.read_file('/home/users/eejasm/SAWS/provinces/Province_New.shp')
    # Remove missing or empty geometries
    province_gdf = province_gdf[province_gdf.geometry.notnull()]
    province_gdf = province_gdf[~province_gdf.geometry.is_empty]

    # Ensure lon/lat CRS
    if province_gdf.crs != "EPSG:4326":
        province_gdf = province_gdf.to_crs(epsg=4326)

    # Plot provinces
    for geom in province_gdf.geometry:
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black')
    
    
    divider = make_axes_locatable(ax)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=precip_colormap,norm=norm), label='Probability',extend='neither',ticks=np.arange(0,1.1,0.1), cax=make_axes_locatable(ax).append_axes("right", size="4%", pad=0, axes_class=maxes.Axes))
    
    date_str = initialisation_time[:8]  # YYYYMMDD
    output_dir = os.path.join(nowcast_fig_file_path, date_str)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_dir+'/'+initialisation_time+'_intensification_'+lead_time+'.png', bbox_inches='tight')

    
# Load HRIT data and convert to numpy array for SII-NowNet input
def load_data(time, hrit_filepath):
    
    # Use fast single glob + in-memory filtering
    base_path =  hrit_file_path+time[:-4]
    all_files = glob(base_path+'/*'+time+'*')

    ir_108_filenames = [f for f in all_files if 'IR_108' in f]
    epi_filenames    = [f for f in all_files if 'EPI' in f]
    pro_filenames    = [f for f in all_files if 'PRO' in f]
    filenames = ir_108_filenames + epi_filenames + pro_filenames

    # Initialize and load only needed channel
    scene = Scene(reader='seviri_l1b_hrit', filenames=filenames)
    scene.load(['IR_108'])

    IR_108 = scene["IR_108"]

    # Projection and coordinate transform
    msg_proj = ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831)
    transformer = Transformer.from_crs("EPSG:4326", msg_proj, always_xy=True)

    lon, lat = 28.0, -26.2
    cen_x, cen_y = transformer.transform(lon, lat)

    # Nearest grid point
    cen_x = IR_108.sel(x=cen_x, y=cen_y, method='nearest').x.item()
    cen_y = IR_108.sel(x=cen_x, y=cen_y, method='nearest').y.item()

    # Find pixel index (faster than meshgrid)
    x_index = np.argmin(np.abs(IR_108.x.values - cen_x))
    y_index = np.argmin(np.abs(IR_108.y.values - cen_y))

    # Crop region
    south_africa_ir = IR_108.isel(x=slice(x_index - 208, x_index + 208), y=slice(y_index - 208, y_index + 208))

    return np.fliplr(south_africa_ir.values)


# normalize input data before insert to SII-NowNet - predefined max/min values
max_val = 327.67
min_val = 177.81
def normalize(array, max_val, min_val):
    array = (array - min_val) / (max_val - min_val + 1e-13)
    return array




# location of you HRIT BT data (IR 10.8um)
hrit_file_path = '/gws/ssde/j25b/swift/WISER-EWSA/HRIT/data/'
# location of the SII-NowNet models
model_file_path = '/home/users/eejasm/SAWS/no_optimizer_models/'
# location to save the nowcast images
nowcast_np_file_path = '/home/users/eejasm/SAWS/nowcast_arrays/'
# location to save the nowcast numpy arrays
nowcast_fig_file_path = '/home/users/eejasm/SAWS/figures/'



# Time argument needs to be in the format to fit the hrit_file_path so that load_data runs correctly (e.g. for me the HRIT data has the time format YYYYMMDDHHMM)
initialisation_time = sys.argv[1]

dt_initialisation_time_minus_1 = datetime.strptime(initialisation_time, '%Y%m%d%H%M') - timedelta(hours=1)
initialisation_time_minus_1 = datetime.strftime(dt_initialisation_time_minus_1, '%Y%m%d%H%M')

input_data = np.dstack([load_data(initialisation_time_minus_1, hrit_file_path), load_data(initialisation_time, hrit_file_path)])
input_data = normalize(input_data, max_val, min_val)[np.newaxis]

inputs = Input(shape=(416, 416, 2))

for lead_time in range(1,4):

    if lead_time == 3:
        outputs = TFSMLayer(filepath=model_file_path+str(lead_time)+'_intensification', call_endpoint='serving_default', trainable=False)(inputs)
        model = Model(inputs, outputs)
        nowcast = model.predict(input_data)[0,:,:,0]
        
        np.save(nowcast_np_file_path+initialisation_time+'_intensification_3.npy', nowcast)
        save_intensification_fig(nowcast, initialisation_time, str(lead_time), nowcast_fig_file_path)
        
    else:
        for init_or_intens in ['initiation', 'intensification']:
            outputs = TFSMLayer(filepath=model_file_path+str(lead_time)+'_'+init_or_intens, call_endpoint='serving_default', trainable=False)(inputs)
            model = Model(inputs, outputs)
            nowcast = model.predict(input_data)[0,:,:,0]

            np.save(nowcast_np_file_path+initialisation_time+'_'+init_or_intens+'_'+str(lead_time)+'.npy', nowcast)
            
            if init_or_intens == 'initiation':
                save_initiation_fig(nowcast, initialisation_time, str(lead_time), nowcast_fig_file_path)
            elif init_or_intens == 'intensification':
                save_intensification_fig(nowcast, initialisation_time, str(lead_time), nowcast_fig_file_path)
                










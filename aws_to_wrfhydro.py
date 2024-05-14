"""
Created on Mon Dec 4 21:24:45 2023

@author: msuarez

Some of the comments in this script are in Spanish
"""

####################################################################################################
####################################################################################################
####################### THE FOLLOWING LIBRARIES NEED TO BE IMPORTED ##############################
import numpy as np
from netCDF4 import Dataset
import wrf
from wrf import (to_np, getvar, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import netCDF4
import datetime
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import geopandas as gpd

####################################################################################################
####################################################################################################
# This script grids the precipitation measured by surface meteorological stations to the WRF-Hydro domain.

# Files needed:
#                                       1) geo_em (geogrid)
#                                       2) csv files with measured rainfall
#                                       3) csv file with the geographic coordinates of the stations

# Before run follow these instructions: 
#                                       1) Modify the start time
#                                       2) Modify the end time
#                                       3) Modify delta time (in general it will be 10 minutes) 
#                                       4) Save the csv files of the stations and locations in the same directory
#                                          where the file ws_to_wrfhydro.py is located 
#                                       5) Modify the path to the geogrid file
#                                       6) Modify the output path. If left as default,
#                                          a FORCING directory will be created in the 
#                                          same directory where the code is executed.
#                                       7) Execute with the following line in terminal: python aws_to_wrfhydro.py 

# The file with the locations of the meteorological stations must be named aws_loc.csv and 
# its content MUST BE in the following format:

# from now on, aws stands for automatic weather station

# number,aws_name,lat,lon
# 1,Rio_Santa_Rosa_Cuenca_Media,-32.03688,-64.63411
# 2,San_miguel_rios,-32.04346,-64.75008

# where the first number must MATCH the name of the aws file.
# Example:
# aws_1.csv
# aws_2.csv
# where aws_1.csv corresponds to the Rio_Santa_Rosa_Cuenca_Media station and aws_2.csv
# corresponds to the San_miguel_rios station

# The content of the aws_x.csv files SHOULD look like the following example:

# date,time,rain
# 25-03-2020,03:20,0
# 25-03-2020,03:30,0.2
# 25-03-2020,03:40,0
# 25-03-2020,03:50,0.2
# 25-03-2020,04:00,0.2
# 25-03-2020,04:10,0.2
# 25-03-2020,04:20,0.4

# Final comments:
#                  1) There may be emas located outside the simulation domain.
#                  2) Check well the coordinates of the emas.
#                  3) Point 2 is very important because there is no control of the
#                     location of the stations with respect to the simulation domain.


# **************************************************************************
# *********** From here start the lines that need to be modified ***********
# **************************************************************************
# Format: YYYYY,MM,DD,HH,MM
# start time
inicio = datetime.datetime(2020, 3, 25, 3, 0)
# end time
fin = datetime.datetime(2020, 3, 25, 8, 0)

# path to the geogrid file
geo_path   = './geo_em.d01.nc'

# path to the output dir
# If the directory does not exist, one will be created
output_dir = './FORCING'

# delta time (interval between station observations)
# If this is changed, modify the conversion from mm to mm/s (line rainrate[:] = (temp_grid[::-1]*6)/3600;)
delta = datetime.timedelta(minutes=10)

# There is a shapefile? If not set to False
shape_file = True 
if shape_file:
    path = "/home/msuarez/Documents/ohmc_workspace/Shapes/"
    shapefile = gpd.read_file(path+'cba_deptos.shp') 

# Plot edge delta in degrees to visualise an area beyond the domain
delta_deg = .2
# **************************************************************************
# ********** This is the end of the lines that need to be changed **********
# **************************************************************************
####################################################################################################
####################################################################################################
# Este colormap es opcional para el grafico de la precipitación
import matplotlib

nws_precip_colors = [
    "#fdfdfd",
    "#04e9e7",
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6"]   # 10.00+

precip_colormap = matplotlib.colors.ListedColormap(nws_precip_colors)
####################################################################################################
####################################################################################################
# Carga del archivo geogrid
geo_file = Dataset(geo_path)
# Elijo la variable altura del terreno
var = wrf.getvar(geo_file, 'HGT_M')
# Extraigo las latitudes y longitudes del archivo geogrid
lats_geofile, lons_geofile = latlon_coords(var)
# Verifico el output de salida si existe o no
# If folder doesn't exist, then create it.
check_output_dir = os.path.isdir(output_dir)
if not check_output_dir:
    os.makedirs(output_dir)
    print("created folder : ", output_dir)
else:
    print(output_dir, "folder already exists.")
####################################################################################################
####################################################################################################
# Defino funciones para la interpolacion
def distance_matrix(x0, y0, x1, y1):
    """
    Calculate distance matrix.
    Note: from <http://stackoverflow.com/questions/1871536>
    """

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    # calculate hypotenuse
    return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi, beta=2):
    """
    Simple inverse distance weighted (IDW) interpolation
    x`, `y`,`z` = known data arrays containing coordinates and data used for interpolation
    `xi`, `yi` =  two arrays of grid coordinates
    `beta` = determines the degree to which the nearer point(s) are preferred over more distant points.
            Typically 1 or 2 (inverse or inverse squared relationship)
    """

    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    # weights = 1.0/(dist+1e-12)**power
    weights = dist ** (-beta)

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    return np.dot(weights.T, z)
####################################################################################################
####################################################################################################
# # size of the grid to interpolate
nx, ny = lats_geofile[0,:].shape[0], lats_geofile[:,0].shape[0]

# generate two arrays of evenly space data between ends of previous arrays
xmin = float(np.min(lons_geofile))
xmax = float(np.max(lons_geofile))
ymin = float(np.min(lats_geofile))
ymax = float(np.max(lats_geofile))

xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)

# generate grid
xi, yi = np.meshgrid(xi, yi)

# colapse grid into 1D
flatten_xi, flatten_yi = xi.flatten(), yi.flatten()
print(50*'*')
print(50*'*')
print('-> Domain dimensions:')
print('west_easth:'+str(nx)+'\nsouth_north:'+str(ny))
print(50*'*')
print(50*'*')
print('\n')
####################################################################################################
####################################################################################################
df_loc = pd.read_csv('./aws_loc.csv')
####################################################################################################
####################################################################################################
print(50*'*')
print(50*'*')
print('-> The raingauge/automatic weather station locations are:')
for index,row in df_loc.iterrows():
    print(row[1],row[2], row[3])
print('-> Domain bounds are:')
print('min_lon:',xmin,' max_lon:',xmax,'\nmin_lat:',ymin,' max_lat:',ymax)
print(50*'*')
print(50*'*')
####################################################################################################
####################################################################################################
# Creo un dict vacio para guardar los registros de las emas
ema_dic = {}

for i in range(len(df_loc)):
  ema_dic['aws_'+str(i+1)] = pd.read_csv('./aws_'+str(i+1)+'.csv')
  ema_dic['aws_'+str(i+1)]['date'] = ema_dic['aws_'+str(i+1)]['date'] +' '+ ema_dic['aws_'+str(i+1)]['time']
  ema_dic['aws_'+str(i+1)] = ema_dic['aws_'+str(i+1)].drop('time', axis=1)
  ema_dic['aws_'+str(i+1)]['date'] =  pd.to_datetime(ema_dic['aws_'+str(i+1)]['date'], format='%d-%m-%Y %H:%M')
  ema_dic['aws_'+str(i+1)].set_index('date', inplace=True)
  ema_dic['aws_'+str(i+1)]['rain'] = ema_dic['aws_'+str(i+1)]['rain'].fillna(0)
####################################################################################################
####################################################################################################
# iterate over range of dates
while (inicio <= fin):
    lat_list = []
    lon_list = []
    precip_list = []
    print('Regridding:'+str(inicio))

    try:
      for i in range(len(df_loc)):
          lat = float(df_loc [df_loc['number'] == i+1]['lat'])
          lon = float(df_loc [df_loc['number'] == i+1]['lon'])
          precip = ema_dic['aws_'+str(i+1)]['rain'].loc[inicio]

          lon_list.append(lon)
          lat_list.append(lat)
          precip_list.append(precip)

      # Calculate IDW
      temp_grid = simple_idw(lon_list, lat_list, precip_list, flatten_xi, flatten_yi, beta=2)
      temp_grid = temp_grid.reshape((ny, nx))

      #################################################################################################
      unout = 'days since 2000-01-01 00:00:00'
      # using netCDF3 for output format
      # YYYYMMDDHHMM.PRECIP_FORCING
      ncout = Dataset(output_dir+'/'+inicio.strftime("%Y")+inicio.strftime("%m")+inicio.strftime("%d")+inicio.strftime("%H")+inicio.strftime("%M")+'.PRECIP_FORCING.nc',
                      'w','NETCDF4_CLASSIC');

      ncout.createDimension('Time', 1);
      ncout.createDimension('DateStrLen', 20);
      ncout.createDimension('south_north', temp_grid[::-1].shape[0]);
      ncout.createDimension('west_east', temp_grid[::-1].shape[1]);

      var_time = ncout.createVariable('Times', 'S1', ('DateStrLen'))
      str_out = netCDF4.stringtochar(np.array([datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")], 'S20'))
      var_time[:] = str_out

      rainrate = ncout.createVariable('precip_rate','float32',('Time','south_north','west_east'))
      rainrate.setncattr('units','mm/s'); rainrate[:] = (temp_grid[::-1]*6)/3600;

      #Add global attributes
      rainrate.remap = "regridded via inverse-distance-weighting method"
      today = datetime.datetime.today()
      ncout.history = "@author: msuarez. Created on " + today.strftime("%d/%m/%y")
      #Add local attributes to variable instances
      rainrate.units = 'mm s^-1'
      rainrate.description = 'RAINRATE'
      rainrate.long_name = 'RAINRATE'

      ncout.close()
      #################################################################################################

      # Create the figure and axis objects
      fig, ax = plt.subplots(1, 1, figsize=(10, 8))

      # Plot the shapefile
      if shape_file:
        shapefile.plot(ax=ax, facecolor="none", edgecolor='lightgrey', zorder=2)

      # Plot the interpolated rainfall data
      contour = ax.contourf(xi, yi, temp_grid, cmap=precip_colormap,#cmap='Reds',
                            levels = np.arange(0,21,1), extend='max')

      for i in range(len(precip_list)):
        #if xmin <= lon_list[i] <= xmax and ymin <= lat_list[i] <= ymax:
        ax.scatter(lon_list[i], lat_list[i], color="black", s=5)
        ax.text(lon_list[i]+.005, lat_list[i]+.005, precip_list[i])

      plt.ylabel('Latitude (S-N)')
      plt.xlabel('Longitude (W-E)')
      plt.title("Squared IDW Interpolation T"+str(inicio))

      cbar = fig.colorbar(contour, ax=ax, shrink=.7, orientation = 'horizontal')
      cbar.set_label('Rainfall [mm]')

      plt.xlim(left=xmin-delta_deg, right=xmax+delta_deg)
      plt.ylim(bottom=ymax+delta_deg, top=ymin-delta_deg)

      plt.tight_layout()
      plt.savefig(output_dir+'/'+str(inicio))
      plt.close('all')

    except KeyError as error:
      print('No record of rainfall:', error)
      continue

    inicio += delta
####################################################################################################
####################################################################################################
print('********************************')
print('********************************')
print('********************************')
print('******* PROCESS FINISHED *******')
print('********************************')
print('********************************')
print('********************************')

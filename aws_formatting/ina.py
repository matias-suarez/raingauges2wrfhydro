#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:27:53 2024

@author: msuarez
"""

import pandas as pd
from datetime import datetime


# path to file
path_to_file = '/home/msuarez/'
# what is the name of the file?
filename = '1560.xls'
# Define start and end datetime (in UTC!!!) in format YYYY-MM-DD HH:MM:SS
# Select start and end minutes multiples of 10
start_datetime = '2020-03-24 23:30:00'
end_datetime = '2020-03-26 13:40:00'

# read the file
df = pd.read_excel(path_to_file+filename)

# Generate datetime range with 10-minute frequency
datetime_range = pd.date_range(start=start_datetime, end=end_datetime, freq='5Min')

# Create DataFrame with datetime column
df_filled = pd.DataFrame({'datetime': datetime_range})

# Add second column with value equal to zero for each row
df_filled['rain'] = 0  # You can assign a scalar value directly
# set index
df_filled.set_index('datetime', inplace=True)

df_temp = pd.DataFrame()
df['Fecha/Hora desde'] =  pd.to_datetime(df['Fecha/Hora desde'], format='%d/%m/%Y %H:%M')
df['Fecha/Hora hasta'] =  pd.to_datetime(df['Fecha/Hora hasta'], format='%d/%m/%Y %H:%M')
df.set_index('Fecha/Hora desde', inplace=True)                                                # set index as the time column
df.index = df.index + pd.Timedelta(hours=3)                                                   # Convert Argentina time to Universal time UTC

df_concatenated = pd.concat([df_filled, df], axis=1)

df_temp = df_concatenated['Valor [mm]'].resample('10Min').sum()

df_out = pd.DataFrame()                                                                       # create an empty dataframe
df_out['rain'] = df_temp                                                                      # save the rain column
df_out.reset_index(inplace=True)                                                              # reset the index
df_out['time'] = df_out['index'].map(lambda i: i.strftime('%H:%M'))                           # create column hour
df_out['date'] = df_out['index'].map(lambda i: i.strftime('%d-%m-%Y'))                        # create column date
df_out = df_out[['date', 'time', 'rain']]                                                     # order columns
df_out.to_csv(path_to_file+filename.split('.')[0]+'.csv', index=False)                        # export to csv
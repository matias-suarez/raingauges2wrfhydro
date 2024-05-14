#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:25:40 2024

@author: msuarez
"""

import pandas as pd
from datetime import datetime


# path to the .xlsx file
archivo = 'ClimaReporte-23-03-2020--30-03-2020.xlsx'
# read the file with pandas
df = pd.read_excel(archivo, sheet_name=None)

# read each sheet
for sheet_name in list(df.keys()):
  df_out = pd.DataFrame()
  print(sheet_name)
  df = pd.read_excel(archivo, skiprows=5, sheet_name=sheet_name)
  # if the sheet has data
  if len(df.index) != 0:
    try:
      df['Fecha'] =  pd.to_datetime(df['Fecha'], format='%d-%m-%Y %H:%M') # convert to datetime object
      df['Fecha'] = df['Fecha'] + pd.Timedelta(hours=3)                   # Convert Argentina time to Universal time UTC
      df_out['rain'] = df['Registro de Lluvia [mm]']                      # create column rain
      df_out['time'] = df['Fecha'].map(lambda i: i.strftime('%H:%M'))     # create column hour
      df_out['date'] = df['Fecha'].map(lambda i: i.strftime('%d-%m-%Y'))  # create column date
      df_out = df_out[['date', 'time', 'rain']]                           # order columns
      df_out.to_csv(sheet_name+'.csv', index=False)          # export to csv
    except Exception as e:
      print('Error: ',e)
      continue
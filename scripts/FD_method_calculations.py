# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:26:16 2021

@author: Usuario
"""

# FD_method_calculations.py
# Version date: 15/11/23

    
   
    
   
#%% 15/11/23  Corrida con restricciones  

import FlashDroughts as fd
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import time
import xarray as xr
import os


#%% Establezco Ruta de archivos:
    
data_path = os.path.join('/Users','Usuario','OneDrive','Documentos','2-DOCTORADO', 'FlashDrought' ,'1-Mundo')
results_path = os.path.join('/Users','Usuario','OneDrive','Documentos','2-DOCTORADO', 'FlashDrought' ,'1-Mundo', '2023-11-revision' , 'Resultado-corrida_restricciones')




#%%
# Set lat, lon and temporal steps variables from ERA5 data. 

# Latitud y longitud del área de estudio
latitud = np.linspace(-60, 80, 281)
longitud = np.linspace(-180, 179.5, 720)

# Rango de latitud y longitud en términos de los arreglos numpy (equivalencia)
pasos_lat = np.arange(0, 281)
pasos_lon = np.arange(0, 720)
np_lon_grid, np_lat_grid = np.meshgrid(pasos_lon, pasos_lat)
np_lat_puntos = np_lat_grid.reshape(202320)
np_lon_puntos = np_lon_grid.reshape(202320)


# Cantidad de pasos temporales en la serie temporal
pasos_tiempo = 5183
tiempo = np.arange(0,pasos_tiempo)
pen_año = int(365/5)


# Forma de los arreglos numpy de las variables
shape = (5183, 281,720)


    
#%% Cálculo eventos

# # cálculo swdi en caso de calcular acá. 
# swdi = fd.calcular_swdi(SMpen, fieldcap, wiltingpoint)
# array_swdi = swdi.filled()
# np.save('swdi.npy', array_swdi)

# Archivo swdi: fue calculado desde matlab se importa como .dat.
fn_variable = data_path + os.path.join('/SWDI_1950-2020.dat')
pen_swdi = fd.leer_variable_matlab(fn_variable, shape)



# Enmascarar océano para hacer más eficiente el cálculo: 
pen_swdi_masked = ma.masked_invalid(pen_swdi) # enmascara los nan o inf ya que en el cálculo de swdi desde matlab el océano tiene valores inf. 

# ahora enmascaro las lat y lon
# Crear una máscara donde pen_swdi tiene valores infinitos
mask = np.isinf(pen_swdi)
# Aplicar la máscara
np_lat_grid_masked = np.ma.masked_array(np_lat_grid, mask[0,:,:])
np_lon_grid_masked = np.ma.masked_array(np_lon_grid, mask[0,:,:])
np_lat_puntos_masked = np_lat_grid_masked.reshape(202320)
np_lon_puntos_masked = np_lon_grid_masked.reshape(202320)

# plt.imshow(pen_swdi[100,:,:], origin = 'lower')
# plt.colorbar()
# plt.imshow(pen_swdi_masked[100,:,:], origin = 'lower')
# plt.colorbar()



# Umbrales: modificar cada vez.
swdi_max = -3
swdi_min = -5
swdi_restriccion = -4

# corrida con todo enmascarado: 
start_time = time.time()
FD_swdi = fd.evaluar_FD_swdi_restricciones(pen_swdi_masked, swdi_max, swdi_min,  np_lat_puntos_masked, np_lon_puntos_masked, pasos_tiempo, shape, swdi_restriccion = -4)
print('--- %s minutes---' % ((time.time()-start_time)/60.))



# Guardo como .nc
# Datos del mundo: uso precipitación para sacar los datos. 
mundo_tp = xr.open_dataset(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc')).tp

# Crear un Data Array 
FD_swdi_nc = xr.DataArray(data=FD_swdi, 
    dims = list(mundo_tp.dims),
    coords = dict(
        lon = mundo_tp.lon,
        lat = mundo_tp.lat,
        time = mundo_tp.time,
    ),
    name = 'FD',
     attrs = dict(
         description = 'Eventos de Flash Drought identificados por método de SWDI con restricción (-4)',
         units = 'adim',
     ),
)   
# Guardo para tener para la próxima
FD_swdi_nc.to_netcdf(results_path + os.path.join('/FD_swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1950-2020.nc'))


#%% Calculo eventos totales desde 1960 
#Cortar desde 1960
FD_swdi_cortado = FD_swdi[730:,:,:]    #uso el FD_swdi array de numpy 
# Cantidad de eventos
#Calcula cantidad de eventos por punto y devuelve array 2D con shape(latitud.size, longitud.size)
cantidad_eventos = FD_swdi_cortado.sum(axis=0, keepdims = False)
# plt.imshow(cantidad_eventos, origin = 'lower')
# plt.colorbar()


#Guardo .nc
cantidad_eventos_nc = xr.DataArray(
    cantidad_eventos,
    coords=[("lat", mundo_tp.lat.values), ("lon", mundo_tp.lon.values)],
    name="eventosFD",
    attrs = dict(
        description = 'Cantidad total de eventos de Flash Drought por cada punto de grilla para los años 1960-2020, identificados por método de SWDI con restricción (-4) ',
        units = 'eventos'
    ),
)
cantidad_eventos_nc.to_netcdf(results_path + os.path.join('/cantidad_eventos_FD-swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1960-2020.nc'))


# guardo enmascarado: 
cantidad_eventos_masked = np.ma.masked_array(cantidad_eventos, mask[0,:,:])
# plt.imshow(cantidad_eventos_masked, origin = 'lower', cmap='viridis_r')
# plt.colorbar()
#Guardo .nc
cantidad_eventos_masked_nc = xr.DataArray(
    cantidad_eventos_masked,
    coords=[("lat", mundo_tp.lat.values), ("lon", mundo_tp.lon.values)],
    name="eventosFD_masked",
    attrs = dict(
        description = 'Cantidad total de eventos de Flash Drought por cada punto de grilla para los años 1960-2020, identificados por método de SWDI con restriccion (-4). El mar está enmascarado según el archivo de SWDI.',
        units = 'eventos'
    ),
)
cantidad_eventos_masked_nc.to_netcdf(results_path + os.path.join('/cantidad_eventos_FD-swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1960-2020_maskedsea.nc'))



# # Crear df esapacial con lat y lon
# lat_lon_size = latitud.size*longitud.size
# lon_grid, lat_grid = np.meshgrid(longitud, latitud)
# df_FD_espacial = pd.DataFrame(columns = ['lat', 'lon', 'eventos_'+str(swdi_max)+str(swdi_min)+'' ])
# df_FD_espacial['lat'] = lat_grid.reshape(lat_lon_size)
# df_FD_espacial['lon'] = lon_grid.reshape(lat_lon_size)
# df_FD_espacial['eventos_'+str(swdi_max)+str(swdi_min)+''] = cantidad_eventos.reshape(lat_lon_size)
# df_FD_espacial.to_csv(results_path + os.path.join('/cantidad_eventos_FD-swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1960-2020.csv'))



#%% Cálculo eventos para distintos umbrales y guardo solo eventos totales en .nc
#%%

# Archivo: desde matlab.
fn_variable = data_path + os.path.join('/SWDI_1950-2020.dat')
pen_swdi = fd.leer_variable_matlab(fn_variable, shape)


# Enmascarar océano para hacer más eficiente el cálculo: 
pen_swdi_masked = ma.masked_invalid(pen_swdi) # enmascara los nan o inf ya que en el cálculo de swdi desde matlab el océano tiene valores inf. 

# ahora enmascaro las lat y lon
# Crear una máscara donde pen_swdi tiene valores infinitos
mask = np.isinf(pen_swdi)
# Aplicar la máscara
np_lat_grid_masked = np.ma.masked_array(np_lat_grid, mask[0,:,:])
np_lon_grid_masked = np.ma.masked_array(np_lon_grid, mask[0,:,:])
np_lat_puntos_masked = np_lat_grid_masked.reshape(202320)
np_lon_puntos_masked = np_lon_grid_masked.reshape(202320)


# Umbrales: modificar cada vez.
swdi_max = -3
swdi_min = -6
swdi_restriccion = -5    # ver cuánto poner

# corrida con todo enmascarado: 
start_time = time.time()
FD_swdi = fd.evaluar_FD_swdi_restricciones(pen_swdi_masked, swdi_max, swdi_min,  np_lat_puntos_masked, np_lon_puntos_masked, pasos_tiempo, shape, swdi_restriccion = -4)
print('--- %s minutes---' % ((time.time()-start_time)/60.))



# Calculo eventos totales desde 1960 
#Cortar desde 1960
FD_swdi_cortado = FD_swdi[730:,:,:]
# Cantidad de eventos
#Calcula cantidad de eventos por punto y devuelve array 2D con shape(latitud.size, longitud.size)
cantidad_eventos = FD_swdi_cortado.sum(axis=0, keepdims = False)


# Guardo como .nc
# Datos del mundo: uso precipitación para sacar los datos. 
mundo_tp = xr.open_dataset(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc')).tp

cantidad_eventos_nc = xr.DataArray(
    cantidad_eventos,
    coords=[("lat", mundo_tp.lat.values), ("lon", mundo_tp.lon.values)],
    name="eventosFD",
    attrs = dict(
        description = 'Cantidad total de eventos de Flash Drought por cada punto de grilla para los años 1960-2020, identificados por método de SWDI con restricción (-4) ',
        units = 'eventos'
    ),
)
cantidad_eventos_nc.to_netcdf(results_path + os.path.join('/cantidad_eventos_FD-swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1960-2020.nc'))


# # Crear df esapacial con lat y lon
# lat_lon_size = latitud.size*longitud.size
# lon_grid, lat_grid = np.meshgrid(longitud, latitud)
# df_FD_espacial = pd.DataFrame(columns = ['lat', 'lon', 'eventos_'+str(swdi_max)+str(swdi_min)+str(swdi_restriccion)+'' ])
# df_FD_espacial['lat'] = lat_grid.reshape(lat_lon_size)
# df_FD_espacial['lon'] = lon_grid.reshape(lat_lon_size)
# df_FD_espacial['eventos_'+str(swdi_max)+str(swdi_min)+str(swdi_restriccion)+''] = cantidad_eventos.reshape(lat_lon_size)
# df_FD_espacial.to_csv(results_path + os.path.join('/cantidad_eventos_FD-swdi_'+str(swdi_max)+str(swdi_min)+'_era5_restriccion_'+str(swdi_restriccion)+'_1960-2020.csv'))





#%% Calcular cantidad de eventos por mes 1960- 2020
#%%

dataset = xr.open_dataset(results_path + os.path.join('/FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))
eventos =  dataset.FD
array_3d_eventosFD_cortado = eventos.where(eventos['time.year']>=1960).dropna('time').values 


shape = array_3d_eventosFD_cortado.shape    #(4453, 281,720)
eventos_total_meses = np.zeros((12,shape[1], shape[2]))

# Indices de meses como diccionario
meses = {
         '1':(0,5),
         '2':(6,11),
         '3':(12,17),
         '4':(18,23),
         '5':(24,29),
         '6':(30,35),
         '7':(36,41),
         '8':(42,48),
         '9':(49,54),
         '10':(55,60),
         '11':(61,66),
         '12':(67,72)
         }

# Correr para todos los meses automatico
for mes in meses:
    eventos_mes = fd.calcular_eventos_mes(mes, meses, shape, array_3d_eventosFD_cortado)   # calculo los eventos para ese mes
    eventos_total_meses[int(mes)-1,:,:] = eventos_mes    # anoto para ese mes en la matriz de salida
np.save(results_path + os.path.join('/cantidad_eventos_por_mes_-3-5_era5_restriccion_-4_1960-2020.npy', eventos_total_meses))  



# verificacion = eventos_total_meses.sum(axis=0, keepdims=True)
# eventos_original = array_3d_eventosFD_cortado.sum(axis=0, keepdims=True)
# verificacion2 = np.nonzero((eventos_original - verificacion) != 0)
# # verificacion2
# # Out[34]: (array([], dtype=int64), array([], dtype=int64), array([], dtype=int64))


#%% Cálculo eventos por mes con xarray

# abro archivo de eventos
dataset = xr.open_dataset(results_path + os.path.join('/FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))
eventos = dataset.FD.where(eventos['time.year']>=1960).dropna('time')   # a partir de 1960

#calculo eventos por mes y obtengo data array
eventos_mes = eventos.groupby('time.month').sum('time')


# Guardar en un Data Array. Tengo que volver a convertir a data array. 
eventos_dataarray = xr.DataArray(data=eventos_mes, 
    dims = list(eventos_mes.dims),
    coords = dict(
        lon = eventos_mes.lon,
        lat = eventos_mes.lat,
        month = eventos_mes.month,
    ),
    name = 'ETM',
     attrs = dict(
         description = 'Eventos totales por mes',
         units = 'adim',
     ),
)   

eventos_dataarray.to_netcdf(results_path + os.path.join('/cantidad_eventos_por_mes_-3-5_era5_restriccion_-4_1960-2020.nc'))



#%% Cálculo de lags.
#%%

dataset = xr.open_dataset(results_path + os.path.join('/FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))
eventos = dataset.FD
eventos = eventos.where(eventos['time.year']>=1960).dropna('time').values
# estos eventos tienen 0 donde está el océano y en lugares donde da 0, así que no necesito enmascarar porque ya la función toma solo los puntos donde hay valores mayores a 0.

shape = eventos.shape   #(4453, 281,720)
# busco indices de eventos
eventos_index = np.nonzero(eventos != 0)  #indices de la matriz de los elementos con eventos
# eventos_index[0].shape
# Out[32]: (294230,)

lags = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
cantidad_lags = len(lags) 

# Variables a correr: 
    
# Humedad de suelo: preprocesamiento: 
variable = 'SM_masked_menor-10'
array_variable_entrada = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_SML123_pentad_anom_world_0.5degrees_1960-2020_nuevo_anomstd.nc'), 'sm')
#prueba = np.nonzero(array_variable_entrada < -20)   # hay un montón de puntos por debajo de -20, se enmascaran. 
array_SM_entrada = ma.masked_where(array_variable_entrada <-10,array_variable_entrada)
  
# SWDI: preprocesamiento 
variable = 'SWDI'
fn_variable = data_path + os.path.join('/SWDI_1950-2020.dat')
array_variable_entrada = fd.leer_variable_matlab(fn_variable, (5183, 281,720))
# Se ven valores infinitos que es el oceano 
array_variable_entrada = array_variable_entrada[730:,:,:]  # corto desde 1960
array_SWDI_entrada = ma.masked_invalid(array_variable_entrada) # enmascara los nan o inf.  
 
  
array_variables = {
        'PR': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1960-2020_365_mm_day_anomstd.nc'), 'tp'), 
        'SM_masked_menor-10': array_SM_entrada,
        'TAS': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_TAS_pentad_world_0.5degrees_1960-2020_standardized_anom_final.nc'), 'tas'),
        'EVT': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_EVT_pentad_anom_world_1960-2020_365_corregido_mm_day_anomstd.nc'), 'ev'),
        'SWDI_masked_undef': array_SWDI_entrada
            } 
    

for variable, array_variable_entrada in array_variables.items():

    # crear matriz para almacenar los lags.
    matriz_lags_variable = np.empty((cantidad_lags,shape[1],shape[2]))   #hay tantas matrices como lags. 
    
    for lag in lags: 
    # Una vez que estas con una sola variable, va CAMBIANDO EL LAG ACÁ:
        variable_lag_prom = fd.crear_matriz_lag(eventos_index, array_variable_entrada, shape, lag)
        variable_lag_prom = variable_lag_prom.filled(fill_value = -999)        
        # archivo de resultados para todos los lags de una variable
         # Serían: -4, -3, -2, -1, 0, 1, 2, 3, 4
        orden_lag = lags.index(lag)    # Entre 0 y cantidad de lags-1 (o sea 8), la ubicación donde queremos guardar el lag
        matriz_lags_variable[orden_lag, :,:] = variable_lag_prom 
    # Una vez guardados los niveles que se quiere guradar, se saca el archivo:  
    matriz_lags_variable_nc = xr.DataArray(data=matriz_lags_variable, 
        dims = ['lags', 'lat', 'lon' ],
        coords = dict(
            lon = dataset.lon,
            lat = dataset.lat,
            lags = np.array(lags),
        ),
        name = f'{variable}',
         attrs = dict(
             description = f'{variable} para los lags {lags}',
             mascara = -999
             
         ),
    )    
    matriz_lags_variable_nc.to_netcdf(results_path + os.path.join(f'/{variable}-{cantidad_lags}lags-FD_swdi_-3-5_era5_restriccion_-4_1960-2020.nc'))




# Guardar variables  en npy y csv. 
# ------------------------------------------

variables = ('PR', 'SM_masked_menor-10', 'TAS', 'EVT', 'SWDI_masked_undef')

#--------------------------------------------------------
# Esto se corre una vez porque es lo mismo para todas las variables
latitud = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc'), 'lat')
longitud = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc'), 'lon')
#lon360 = np.linspace(180, 359.5, 360)
#longitud[0:360] = lon360
nombres_columnas = ['lat', 'lon', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4']
land_sea_mask = False
lags = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


# guardar en csv y npy
for variable in variables: 

    matriz_lags_variable = xr.open_dataset(results_path + os.path.join(f'/{variable}-9lags-FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))
    
    # guardar npy
    np.save(results_path + os.path.join(f'/{variable}-9lags-FD_swdi_-3-5_era5_restriccion_-4_1960-2020.npy'),  matriz_lags_variable[variable].values)
    
    
    # Guardar en csv cada lag de cada variable con función nueva y con máscara        
    # Correr esto para cada variable distinta
    
    df_lags_variable = fd.crear_df_espacial(latitud, longitud, nombres_columnas, land_sea_mask, mascara = False)
    for lag in lags: 
        df_lags_variable[str(lag)] = matriz_lags_variable[variable].values[lags.index(lag),:,:].reshape(latitud.size*longitud.size)
    df_lags_variable.to_csv(results_path + os.path.join(f'/matriz_{variable}_9lags.csv'))
    
    
    




#%% # Calcular Ventanas lags. 21-11-23
#%%


import FlashDroughts as fd
import numpy as np
import numpy.ma as ma
import time
import xarray as xr



#%% Armar máscara de hotspots:
# Establecer filtros por región crítica: 
# Datos hotspots: nombre: (lon_O, lon_E, lat_N, lat_S)    
hotspots = {
         'SESA':(-61,-50, -23, -33),
         'NSA':(-77,-66, 12, 3),
         'CEUSA':(-100,-70, 47, 29),
         'SCh':(105, 122, 35, 21),
         'SEAs':(95, 108, 19, 0),
         'In':(72, 83, 32, 15),
         'CEEu':(2, 44, 55, 46),
         'SRus':(44, 87, 58, 49),
         'CWAf':(-16, 1, 15, 8),
         }


latitud = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc'), 'lat')
longitud = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1950-2020_365_mm_day.nc'), 'lon')
lon_grid, lat_grid = np.meshgrid(longitud, latitud)
lat_grid = lat_grid.filled()
lon_grid = lon_grid.filled()
mascara_hotspots = np.zeros((lat_grid.shape)) - 999
for hotspot in hotspots: 
    mascara_hotspots[(lon_grid >= hotspots[hotspot][0]) & (lon_grid <= hotspots[hotspot][1]) & (lat_grid <= hotspots[hotspot][2]) & (lat_grid >= hotspots[hotspot][3]) ] = 0
np.save(results_path + os.path.join('/mascara_hotspots_mundo.npy'), mascara_hotspots) 
    
   
#%% Corrida cálculo de ventanas 

# Seleccionar lugar puede ser en hotspots o todo el mundo
lugar = 'hotspots'   # o mundo. 

# Eventos
dataset = xr.open_dataset(data_path + os.path.join('/FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))
eventos = dataset.FD.where(dataset.FD['time.year']>=1960).dropna('time').values     #Cortar desde 1960


#Armar máscara de agua con archivo de eventos sin restricciones
# En caso de no tener ese archivo para armar la máscara, se puede armar la máscara a partir de los valores inf en SWDI (ver más arriba). 
array_3d_eventosFD = np.load(data_path + os.path.join('/FD_SWDI_ERA5_-3_-5_sinrestriccion.npy'))
array_3d_eventosFD = array_3d_eventosFD[730:,:,:]  #Cortar desde 1960
shape = (4453, 281,720)
eventos_totales = array_3d_eventosFD.sum(axis=0)
mascara = np.broadcast_to(eventos_totales, shape)
# aplico máscara agua para enmascarar océano
array_3d_eventosFD_masked = ma.masked_where(mascara == 538, eventos)  # ojooo al ser desde 1960 es 538 el agua y no 625

    
 
if lugar == 'hotspots':
    # # acá enmascaro para dejar sólo los hotspots.
    hotspots = np.load(results_path + os.path.join('/mascara_hotspots_mundo.npy'))
    mascara_hotspots = np.broadcast_to(hotspots, shape)
    array_3d_eventosFD_masked = ma.masked_where(mascara_hotspots == -999,array_3d_eventosFD_masked)


# corro con las máscaras que haya aplicado
eventos_index = np.nonzero(array_3d_eventosFD_masked != 0)  #indices de la matriz de los elementos con eventos y solo tierra


# Variables a correr: 
 
# Precipitación: preprocesamiento: 
variable = 'PR_masked_menor-10'
array_variable_entrada = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1960-2020_365_mm_day_anomstd.nc'), 'tp') 
#prueba = np.nonzero(array_variable_entrada < -20)   # hay un montón de puntos por debajo de -20, voy a enmascarar. 
array_PR_entrada = ma.masked_where(array_variable_entrada <-10,array_variable_entrada)
  

# Humedad de suelo: preprocesamiento: 
variable = 'SM_masked_menor-10'
array_variable_entrada = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_SML123_pentad_anom_world_0.5degrees_1960-2020_nuevo_anomstd.nc'), 'sm')
#prueba = np.nonzero(array_variable_entrada < -10)   # hay un montón de puntos por debajo de -10, voy a enmascarar. 
array_SM_entrada = ma.masked_where(array_variable_entrada <-10,array_variable_entrada)
  
# SWDI: preprocesamiento 
variable = 'SWDI'
fn_variable = data_path + os.path.join('/SWDI_1950-2020.dat')
array_variable_entrada = fd.leer_variable_matlab(fn_variable, (5183, 281,720))
array_variable_entrada = array_variable_entrada[730:,:,:] 
array_SWDI_entrada = ma.masked_invalid(array_variable_entrada) # enmascara los nan o inf.  
 
  
array_variables = {
        'PR_masked_menor-10': array_PR_entrada,
        'SM_masked_menor-10': array_SM_entrada,
        'TAS': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_TAS_pentad_world_0.5degrees_1960-2020_standardized_anom_final.nc'), 'tas'),
        'EVT': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_EVT_pentad_anom_world_1960-2020_365_corregido_mm_day_anomstd.nc'), 'ev'),
        'SWDI_masked_undef': array_SWDI_entrada
            } 
    

# Lags y algunas configuraciones por variable, ojo!
lags = [-4,-3,-2, -1, 0, 1, 2, 3, 4]
cantidad_lags = len(lags) 
w_size = 25  # tamaño del lado de la ventana, debe ser cuadrada y el tamaño de lado deber ser impar. 
window_variable_lags = np.zeros((len(lags), w_size,w_size))
start_time = time.time()



for variable, array_variable_entrada in array_variables.items():

    array_variable_entrada = array_PR_entrada    
    for i,lag in enumerate(lags): 
        print(f'lag {lag}')
        window_variable_lags[i] = fd.crear_ventana_lag(eventos_index, array_variable_entrada, shape, lag, w_size)
        
    print('--- %s minutes---' % ((time.time()-start_time)/60.))
    
    np.save(results_path + os.path.join(f'/Lags_{variable}_AnomaliasStd_ventana{w_size}_{lugar}_{cantidad_lags}-lags.npy'), window_variable_lags)
    






#%% # Calcular Ventanas lags. ALEATORIO 22-11-23
#%%


import FlashDroughts as fd
import numpy as np
import numpy.ma as ma
import time
import xarray as xr



lugar = 'hotspots'   # o mundo. 

# Creo npy con Eventos aleatorios
shape = (4453, 281,720)
eventos_aleatorios = np.random.rand(shape[0], shape[1], shape[2])    # armo eventos aleatorios entre 0 y 1, todos misma probabilidad de ocurrencia. 


#Armar máscara de agua con archivo de eventos sin restricciones
# En caso de no tener ese archivo para armar la máscara, se puede armar la máscara a partir de los valores inf en SWDI (ver más arriba). 
array_3d_eventosFD = np.load(data_path + os.path.join('/FD_SWDI_ERA5_-3_-5_sinrestriccion.npy'))
array_3d_eventosFD = array_3d_eventosFD[730:,:,:]  #Cortar desde 1960
shape = (4453, 281,720)
eventos_totales = array_3d_eventosFD.sum(axis=0)
mascara = np.broadcast_to(eventos_totales, shape)

# aplico máscara agua
array_3d_eventosFD_masked = ma.masked_where(mascara == 538, eventos_aleatorios)  # ojo al ser desde 1960 es 538 el agua y no 625



if lugar == 'hotspots':
    # # acá enmascaro para dejar sólo los hotspots.
    hotspots = np.load(results_path + os.path.join('/mascara_hotspots_mundo.npy'))
    mascara_hotspots = np.broadcast_to(hotspots, shape)
    array_3d_eventosFD_masked = ma.masked_where(mascara_hotspots == -999,array_3d_eventosFD_masked)


# corro con las más caras que haya aplicado
eventos_index = np.nonzero(array_3d_eventosFD_masked > 0.995)  #indices de la matriz de los elementos con eventos aleatorios mayor a 0.9 por ejemplo, voy ajustando según cantidad de eventos que quiero que entren. 


# Variables a correr: 
 
# Precipitación: preprocesamiento: 
variable = 'PR_masked_menor-10'
array_variable_entrada = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_PR_pentad_anom_world_0.5degrees_1960-2020_365_mm_day_anomstd.nc'), 'tp') 
#prueba = np.nonzero(array_variable_entrada < -20)   # hay un montón de puntos por debajo de -20, voy a enmascarar. 
array_PR_entrada = ma.masked_where(array_variable_entrada <-10,array_variable_entrada)
  

# Humedad de suelo: preprocesamiento: 
variable = 'SM_masked_menor-10'
array_variable_entrada = fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_SML123_pentad_anom_world_0.5degrees_1960-2020_nuevo_anomstd.nc'), 'sm')
#prueba = np.nonzero(array_variable_entrada < -10)   # hay un montón de puntos por debajo de -10, voy a enmascarar. 
array_SM_entrada = ma.masked_where(array_variable_entrada <-10,array_variable_entrada)
  
# SWDI: preprocesamiento 
variable = 'SWDI'
fn_variable = data_path + os.path.join('/SWDI_1950-2020.dat')
array_variable_entrada = fd.leer_variable_matlab(fn_variable, (5183, 281,720))
array_variable_entrada = array_variable_entrada[730:,:,:] 
array_SWDI_entrada = ma.masked_invalid(array_variable_entrada) # enmascara los nan o inf.  
 
  
array_variables = {
        'PR_masked_menor-10': array_PR_entrada,
        'SM_masked_menor-10': array_SM_entrada,
        'TAS': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_TAS_pentad_world_0.5degrees_1960-2020_standardized_anom_final.nc'), 'tas'),
        'EVT': fd.leer_variable_netcdf4(data_path + os.path.join('/ERA5_EVT_pentad_anom_world_1960-2020_365_corregido_mm_day_anomstd.nc'), 'ev'),
        'SWDI_masked_undef': array_SWDI_entrada
            } 
    

# Lags y algunas configuraciones por variable, ojo!
lags = [-4,-3,-2, -1, 0, 1, 2, 3, 4]
cantidad_lags = len(lags) 
w_size = 25  # tamaño del lado de la ventana, debe ser cuadrada y el tamaño de lado deber ser impar. 
window_variable_lags = np.zeros((len(lags), w_size,w_size))
start_time = time.time()



for variable, array_variable_entrada in array_variables.items():

    array_variable_entrada = array_PR_entrada    
    for i,lag in enumerate(lags): 
        print(f'lag {lag}')
        window_variable_lags[i] = fd.crear_ventana_lag(eventos_index, array_variable_entrada, shape, lag, w_size)
        
    print('--- %s minutes---' % ((time.time()-start_time)/60.))
    #np.save(f'1-Mundo/2023-06-corrida/Lags_{variable}_AnomaliasStd_ventana{w_size}_mundo.npy', window_variable_lags)
    np.save(results_path + os.path.join(f'/Lags_{variable}_AnomaliasStd_ventana{w_size}_{lugar}_{cantidad_lags}-lags_eventos_aleatorios.npy'), window_variable_lags)
    




#%% # Sacar datos de algunos años y en algunas zonas - 27/11/23
#%%  

#variables = ('USA', 'China', 'India' ,  'RusUk')

# regiones: (lon, lon, lat, lat)

regions = {         
         'USA':(-126,-66, 24, 50), 
         'China':(72, 136, 17, 55),
         'India':(67.5, 98, 7, 36),
         'RusUk':(29, 58, 45, 58)
         }

time = {'USA': '2012', 
        'China':'2013',
        'India':'2001',
        'RusUk':'2010'
        }

meses = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']


#var = 'USA'
dataset = xr.open_dataset(results_path + os.path.join('/FD_swdi_-3-5_era5_restriccion_-4_1950-2020.nc'))


for var in regions: 
    #extraigo región
    regionFD = dataset.FD.sel(lon=slice(regions[var][0], regions[var][1]), lat = slice(regions[var][2], regions[var][3]))
    #extraigo año y resampleo a mensual
    regionFD = regionFD.sel(time=time[var]).groupby('time.month').sum('time')    # uso gropuby porque es un año nomás y así me quedan los meses como 1,2,3....,12. 
    #guardo en csv: 
    # Crear df esapacial con lat y lon
    lat_lon_size = regionFD.lat.size*regionFD.lon.size
    lon_grid, lat_grid = np.meshgrid(regionFD.lon, regionFD.lat)
    df_FD_espacial = pd.DataFrame(columns = ['lat', 'lon', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    df_FD_espacial['lat'] = lat_grid.reshape(lat_lon_size)
    df_FD_espacial['lon'] = lon_grid.reshape(lat_lon_size)
    for mes in meses: 
        df_FD_espacial[mes] = regionFD.sel(month=int(mes)).values.reshape(lat_lon_size)
    df_FD_espacial.to_csv(results_path + os.path.join(f'/Eventos_por_mes-Region_{var}-año_{time[var]}.csv'))




#%% 2024-02-23 cálculo de ciclo anual de SWDI
#%%

swdi = xr.open_dataarray(results_path + os.path.join('/SWDI_mundo_ERA5_1950-2020.nc'))    # es el mismo archivo de SWDI que veniamos usando solo que lo transformé a nc como dataarray. 


# OPCIÓN 2:me genera 73 groups with labels 1, 2, 3, 4, 5, 6, ..., 69, 70, 71, 72, 73.
# Crear una nueva coordenada 'pentada'
swdi['pentada'] = (swdi.time.dt.dayofyear - 1) // 5 + 1

# enmascarar inf
swdi = swdi.where(np.isfinite(swdi))

# Calcular el ciclo anual climatológico
ciclo_anual = swdi.groupby('pentada').mean('time')
ciclo_anual.to_netcdf(results_path + os.path.join('/ciclo_anual-swdi-mundo-ERA5-1950_2020.nc'))


# #Verificación punto -30 lat -60 lon)
# np.savetxt(results_path + os.path.join('/Verificacion/swdi-punto_control-30_-60.txt'), swdi.sel(lat=-30, lon=-60))
# np.savetxt(results_path + os.path.join('/Verificacion/ciclo_anual-swdi-punto_control-30_-60.txt'), ciclo_anual.sel(lat=-30, lon=-60))





























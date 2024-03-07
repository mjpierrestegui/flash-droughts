# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:26:16 2021

@author: Usuario
"""

# FlashDroughts.py (import as fd)
# Version date: 15/11/23



#%%

import netCDF4 as nc
import numpy as np
import pandas as pd
import numpy.ma as ma
#from numba import jit




def leer_variable_netcdf4(fn_variable, variable):
    '''Lee el archivo .nc donde está la variable, y recupera el valor de la 
    varibale como un arreglo de numpy. 
    Pre: tanto fn_variable, como variable deben ser cadenas de texto (str)
         fn_variable debe ser un archivo .nc de tipo netcdf4
    Post: arreglo de numpy'''
   
    ds_variable = nc.Dataset(fn_variable)  #Lee archivo netcdf4 y obtiene DataSet
    #Acceder a los datos de la variable como arreglo de numpy
    array_variable = ds_variable[variable][:]
    
    return array_variable

def leer_variable_matlab(fn_variable, shape):
    '''Lee el archivo .dat donde está la variable, y recupera el valor de la 
    varibale como un arreglo de numpy con la forma indicada.  
    Pre: nombre del archivo fn_variable debe ser cadena de texto (str),
         la forma del arreglo de salida shape debe ser una tupla de tres números.         
    Post: arreglo de numpy con la forma indicada'''
    
    array_variable = np.fromfile(fn_variable, dtype=np.float32).reshape(shape)
    
    return array_variable


 
def calcular_swdi(SMpen, fieldcap, wiltingpoint):
    '''Calcula el índice swdi a partir de pentadas de SM.
    Pre: SMpen y los archivos de fieldcap y wiltingpoint tienen que estar
    recortados para las mismas lat y lon'''
    
    swdi = ((SMpen - fieldcap)/(fieldcap - wiltingpoint)) * 10
    return swdi




def evaluar_FD_swdi_restricciones(pen_swdi, swdi_max, swdi_min,  np_lat_puntos, np_lon_puntos, pasos_tiempo, shape, swdi_restriccion = -4):
    '''Evalua para cada serie tempral de los puntos de grilla (con o sin mascara)
    aquellos que cumplen con condicion de FD: pasar de SWDI mayor o igual 
    a swdi_max, a SWDI menor o igual a swdi_min en 4 pentadas o menos. Además, no puede
    volver a subir a swdi_max en el proceso de bajada. 
    Restricción: En este caso se agrega restricción para declarar FD: el SWDI se tiene que mantener por encima de swdi_restriccion
    durante las 3 pentadas anteriores. 
    Pre: matriz con las series temporales de swdi de pentadas para el área. 
         Parámetros: swdi_max, swdi_min umbrales máximo y mínimo para la versión de FD que se quiera. 
    Post: matriz con 1 para los eventos identificados y 0 cuando no hay eventos.
         El evento se identifica con 1 en la primer pentada del on-set. 
         El método no mide duración del evento ni cantidad de pentadas para 
         llegar a menor o igual a swdi_min
    '''
           
    FD_swdi = np.zeros(shape, dtype = np.int32) 
    fuera_de_indice = 0
    control = 0
    
    
    for np_lat, np_lon in zip(np_lat_puntos, np_lon_puntos):  
        print(np_lat,np_lon)
        for tiempo in(range(pasos_tiempo)):
            try: 
                if pen_swdi[tiempo, np_lat, np_lon] >= swdi_max and ( pen_swdi[tiempo+1, np_lat, np_lon] >= swdi_max or pen_swdi[tiempo+2, np_lat, np_lon] >= swdi_max):
                    FD_swdi[tiempo, np_lat, np_lon] = 0                    
               
                elif pen_swdi[tiempo, np_lat, np_lon] >= swdi_max and ( pen_swdi[tiempo+1, np_lat, np_lon]<= swdi_min or  pen_swdi[tiempo+2, np_lat, np_lon]<= swdi_min or  pen_swdi[tiempo+3, np_lat, np_lon] <= swdi_min):
                    # si se cumple condición para que sea FD, chequeo primero la restricción antes de declararla FD: uso and porque se tiene que dar en todas las pentadas previas. 
                    if pen_swdi[tiempo-1, np_lat, np_lon] > swdi_restriccion and pen_swdi[tiempo-2, np_lat, np_lon] > swdi_restriccion and pen_swdi[tiempo-3, np_lat, np_lon] > swdi_restriccion:
                        # si se cumple la restricción, recién ahí lo declaro FD. Sino no. 
                        FD_swdi[tiempo, np_lat, np_lon] = 1
                        control += 1
                                                
                else: 
                    FD_swdi[tiempo, np_lat, np_lon] = 0
            except IndexError:
                fuera_de_indice += 1 
    print(f' Cantidad de veces fuera de indice = {fuera_de_indice}, Cantidad eventos totales = {control}')
    return FD_swdi





def crear_df_espacial(latitud, longitud, nombres_columnas, land_sea_mask, mascara = False):
    '''Creación de DataFrame en Pandas con info de la latitud y longitud para cada punto de
        grilla, y solo los nombres de las columnas para las variables, para ser llenadas luego.
        Pre: array de Numpy de 2D (shape = (latitud.size, longitud.size) es decir
                                  según la cantidad de pasos de lat y lon, donde la
                                  latitud y longitud son arreglos de npy con los valores
                                  reales de latitud y longitud)
            nombres_columnas debe ser una lista con los nombres de las columnas como 
            strings y debe incluir si o si los nombres lat y lon. Si máscara == True, 
            debe incluir el nombre 'mask'. 
            máscara agua, subir array 2D con la máscara original.          
        Post: genera un archivo csv con DF de tres columnas minimo: latitud, longitud, 
        demás variables espaciales.      
        Por defecto: sin máscara de agua'''
    
    lon_grid, lat_grid = np.meshgrid(longitud, latitud)
    df_FD_espacial = pd.DataFrame(columns = nombres_columnas)
    df_FD_espacial['lat'] = lat_grid.reshape(latitud.size*longitud.size)
    df_FD_espacial['lon'] = lon_grid.reshape(latitud.size*longitud.size)
    
    if mascara == True: 
        df_FD_espacial['mask'] = land_sea_mask.reshape(latitud.size*longitud.size)
   
    return df_FD_espacial




  
#@jit
def crear_matriz_lag(eventos_index, array_variable_entrada, shape, lag):
    '''Crea una matriz de salida que contiene los valores de las variables de 
    entrada en determinados lags y promedidadas en la coordenada donde hay eventos.
    Pre: eventos_index es una tupla de arrays con los índices de np donde hay
    eventos.
          array_variable_entrada: matriz 3D de forma "shape" que contiene el valor 
          de la variable de interes.
          shape: Dimensión de las matrices de entrada formada por tupla de tres números
          enteros. 
          lag: número entero que indica el lag que nos interesa sacar, será positivo
          si el lag es para adelante, y negativo si es para atrás en el tiempo.
    Post: Matriz 3D de solo un nivel en z que tiene el promedio de la variable
    en el lag dado solo para las coordenadas con eventos, en las coordenadas sin 
    eventos va -999.
          '''
    
    coordenadas_eventos = list(zip(eventos_index[0], eventos_index[1], eventos_index[2])) #lista donde cada elemento es la tupla de coordenadas del punto
    matriz_variable_lag = np.zeros(shape)
    matriz_variable_lag[::]= -999  # inicializo en -999 para no confundir valores que den cero
    fuera_serie = 0  # para control
    #matriz_salida = np.empty((1,shape[1],shape[2]))   
    #matriz_salida[:] = -999   # inicializo en -999 para no confundir valores que den cero
    for coordenada in coordenadas_eventos:  
        #recorro todas las coordenas y pasos temporales que tienen eventos 
        if (coordenada[0]+lag)> 0:
            try:   #para evitar error cuando se va de serie el lag
                coordenada_lag = (coordenada[0]+lag, coordenada[1], coordenada[2]) #suma al paso temporal, el lag que se quiere para formar la nueva coordenada
                valor_variable_lag = array_variable_entrada[coordenada_lag]
                matriz_variable_lag[coordenada] = valor_variable_lag
            except IndexError:
                fuera_serie += 1
        else:
          fuera_serie += 1 
         
    print(f'Total fuera de la serie: {fuera_serie}')   # para control
    # Ahora hago el promedio sólo de los valores que hay
    matriz_variable_lag = ma.masked_where(matriz_variable_lag == -999, matriz_variable_lag, copy = False)
    matriz_variable_lag = matriz_variable_lag.mean(axis=0, keepdims = True)
    
    
    return matriz_variable_lag


def calcular_eventos_mes(mes, meses, shape, array_3d_eventosFD):
    '''Calcula la cantidad de eventos que hay en un determinado mes.
       Pre: array 3D con los eventos identificados con 1 y donde no hay eventos 0, siendo
            z la dimensión con los pasos temporales, y el resto lat, lon.
            mes: numero entero del 1 al 12 que identifica el mes que queremos sacar. 
            meses: diccionario que indica en la serie de datos, los indices de 
            comienzo y fin para ese mes en el año 1 de la serie (ejemplo 6 pentadas 
            cada mes, excepto agosto que tiene 7, entonces enero va del 0 al 5)
            shape: la forma que tiene la matriz 3D de entrada. 
       Post: matriz de un nivel en z que tiene la suma de eventos para ese mes
       para cada par lat, lon.
       
           '''
    comienzo = meses[str(mes)][0]
    fin = meses[str(mes)][1]
    años = 0 # para verificar
    eventos_mes = np.zeros((1, shape[1], shape[2]))
    # correr para todos los años de ese dado 'mes'
    while fin < shape[0]:    # menor a la dimension en z porque la dimension tiene un numero mas que el indice maximo, porque se arranca en 0
        suma_parcial = array_3d_eventosFD[comienzo:fin+1,:,:].sum(axis=0, keepdims=True)
        eventos_mes = eventos_mes + suma_parcial
        comienzo += 73
        fin += 73
        años +=1
    
    print(f'Para el mes {mes} dieron {años} años')

    return eventos_mes





#@jit
def crear_ventana_lag(eventos_index, array_variable_entrada, shape, lag, w_size):
    '''
          '''
    
    coordenadas_eventos = list(zip(eventos_index[0], eventos_index[1], eventos_index[2])) #lista donde cada elemento es la tupla de coordenadas del punto
    window_variable_lag = np.zeros((w_size,w_size))
    w_radio = w_size//2
    n_eventos = eventos_index[0].shape
    n_eventos_lag = 0
    fuera_serie = 0  # para control
    #matriz_salida = np.empty((1,shape[1],shape[2]))   
    #matriz_salida[:] = -999   # inicializo en -999 para no confundir valores que den cero
    for coordenada in coordenadas_eventos:  
        #recorro todas las coordenas y pasos temporales que tienen eventos 
        if (coordenada[0]+lag)> 0 and coordenada[1]>=w_radio and coordenada[1]<shape[1]-w_radio-1 and coordenada[2]>=w_radio and coordenada[2]<shape[2]-w_radio-1:
            try:   #para evitar error cuando se va de serie el lag
                #coordenada_lag = [coordenada[0]+lag, coordenada[1]-5:coordenada[1]+6, coordenada[2]-5:coordenada[2]+6] #suma al paso temporal, el lag que se quiere para formar la nueva coordenada
                n_eventos_lag += 1
                window_variable_aux = array_variable_entrada[coordenada[0]+lag, coordenada[1]-w_radio:coordenada[1]+w_radio+1, coordenada[2]-w_radio:coordenada[2]+w_radio+1]
                #print(f'forma variable aux= {window_variable_aux.shape}')
                #print(f'coord = {coordenada}')
                window_variable_lag = window_variable_lag + window_variable_aux.filled(fill_value=0)
            except IndexError:
                fuera_serie += 1
                #print(f'fuera de la serie: {fuera_serie}')
        else:
          fuera_serie += 1 
          
         
    print(f'Total fuera de la serie: {fuera_serie}')   # para control
    print(f'Total de eventos por lag: {n_eventos_lag}, y eventos totales {n_eventos}')
    # Ahora hago el promedio sólo de los valores que hay
    window_variable_lag /= n_eventos_lag
    
    
    
    return window_variable_lag




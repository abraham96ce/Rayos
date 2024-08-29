# Xarray es una biblioteca para trabajar con arreglos multidimensionales etiquetados,
# especialmente útil para manipular datos climáticos y geoespaciales.
!pip install xarray

# Contextily facilita la adición de mapas base a gráficos de Matplotlib, especialmente útil para visualización de datos geoespaciales.
!pip install contextily

# Geopandas extiende la funcionalidad de Pandas para hacer más fácil el trabajo con datos geoespaciales mediante la integración de geometrías y operaciones espaciales.
!pip install geopandas

# Este comando actualiza las bibliotecas Matplotlib y Geopandas en tu entorno de Python para acceder a las últimas mejoras y correcciones de errores en la visualización de datos y manipulación de datos geoespaciales
!pip install --upgrade matplotlib geopandas

# Matplotlib es una biblioteca de visualización en 2D y 3D en Python. Es ampliamente utilizado para crear gráficos y visualizaciones.
!pip install --upgrade matplotlib

# Cartopy es una biblioteca para la visualización de datos geográficos. Proporciona herramientas para proyecciones de mapas y transformaciones geoespaciales.
!pip install cartopy

# Este módulo forma parte de Matplotlib y proporciona herramientas para crear gráficos con ejes divididos y ubicación personalizada.
!pip install mpl_toolkits.axes_divider

# Pandas es una biblioteca para manipulación y análisis de datos. Proporciona estructuras de datos como DataFrame para trabajar con datos tabulares.
!pip install pandas

####################################################################################################################################################
# Monta Google Drive en la ruta "/content/drive", facilitando el acceso a archivos almacenados en Google Drive desde el entorno de Colab.
from google.colab import drive
drive.mount('/content/drive')
####################################################################################################################################################

####################################################################################################################################################
# Se usa la biblioteca xarray y se renombra como xr, para facilitar la manipulación y análisis de datos multidimensionales, especialmente datos climáticos y geoespaciales almacenados en formatos como NetCDF.
# Proporciona estructuras de datos etiquetadas y funciones que simplifican la indexación, selección y manipulación de datos en arreglos multidimensionales, siendo útil para trabajar con conjuntos de datos complejos en ciencia de datos y geociencias.
import xarray as xr

# Ruta al archivo NetCDF
file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300400_e20231002301000_c20231002301018.nc"
#file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300200_e20231002300400_c20231002300417.nc"

# Abre el conjunto de datos
ds = xr.open_dataset(file_path)

# Accede a las variable
event_energy = ds['event_energy']  # Energía asociada con el evento de rayo

# Imprime el número total de eventos de rayo
total_events = len(event_energy)
print(f"\nNúmero total de eventos de rayo: {total_events}")

# Cierra el dataset después de usarlo
ds.close()

####################################################################################################################################################

####################################################################################################################################################
# Se usa la biblioteca xarray y se renombra como xr, para facilitar la manipulación y análisis de datos multidimensionales, especialmente datos climáticos y geoespaciales almacenados en formatos como NetCDF.
# Proporciona estructuras de datos etiquetadas y funciones que simplifican la indexación, selección y manipulación de datos en arreglos multidimensionales, siendo útil para trabajar con conjuntos de datos complejos en ciencia de datos y geociencias.
import xarray as xr

# Ruta al archivo NetCDF
file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300000_e20231002300200_c20231002300216.nc"

# Carga el archivo NetCDF utilizando xarray
ds = xr.open_dataset(file_path)

# Accede a la variable "event_time_offset"
event_time_offset = ds['event_time_offset']

# Muestra los valores de "event_time_offset"
print("Valores de event_time_offset:")
print(event_time_offset.values[:100])


# Cierra el dataset después de usarlo
ds.close()
####################################################################################################################################################

####################################################################################################################################################
import xarray as xr

# Ruta al archivo NetCDF
file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300000_e20231002300200_c20231002300216.nc"

# Carga el archivo NetCDF utilizando xarray
ds = xr.open_dataset(file_path)

# Accede a la variable "event_time_offset"
event_energy = ds['event_energy']

# Muestra los valores de "event_time_offset"
print("Valores de event_energy:")
print(event_energy.values[:100])

# Cerrar el dataset después de usarlo
ds.close()
####################################################################################################################################################

####################################################################################################################################################
import xarray as xr

# Ruta al archivo NetCDF
file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300000_e20231002300200_c20231002300216.nc"

# Carga el archivo NetCDF utilizando xarray
ds = xr.open_dataset(file_path)

# Crea una nueva variable 'group' basada en las coordenadas de latitud
ds['group'] = xr.where(ds['event_lat'] < 0, 'Group 1', 'Group 2')

# Aplica groupby para dividir el dataset en grupos
grouped_ds = ds.groupby('group')

# Itera sobre los grupos
for group, group_ds in grouped_ds:
    # Accede a las variables "event_lon" y "event_lat" dentro del grupo
    group_event_lon = group_ds['event_lon']
    group_event_lat = group_ds['event_lat']

    # Muestra los valores de "event_lat" y "event_lon" dentro del grupo
    print(f"\nValores de event_lat en {group}:")
    print(group_event_lat.values[:100])

    print(f"\nValores de event_lon en {group}:")
    print(group_event_lon.values[:100])

# Cierra el dataset después de usarlo
ds.close()
####################################################################################################################################################

####################################################################################################################################################
# Importa las bibliotecas necesarias
import xarray as xr  # Para el manejo de datos en formato NetCDF
import matplotlib.pyplot as plt  # Para la creación de gráficos
import geopandas as gpd  # Para trabajar con datos geoespaciales
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # Para crear un minimapa en el gráfico
import numpy as np  # Para operaciones numéricas

# Ruta al archivo NetCDF
file_path = "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300000_e20231002300200_c20231002300216.nc"

# Carga el archivo NetCDF utilizando xarray
ds = xr.open_dataset(file_path)

# Accede a las variables relacionadas con la caída de los relámpagos y el tiempo
event_lat = ds['event_lat']  # Latitud geográfica del evento de rayo
event_lon = ds['event_lon']  # Longitud geográfica del evento de rayo
event_time_offset = ds['event_time_offset']  # Tiempo en segundos desde el inicio de la captura del evento de rayo
event_energy = ds['event_energy']  # Energía asociada con el evento de rayo

# Calcula el tiempo real de caída en minutos como valores numéricos
event_time_minutes = event_time_offset.astype(int) / 60.0  # Convertir a tipo int y luego a minutos

# Calcula el tiempo transcurrido desde la caída en minutos
current_time = event_time_minutes.max()  # Tiempo actual (último evento como referencia)
time_since_fall = current_time - event_time_minutes  # Tiempo transcurrido desde la caída

# Se Obtiene los límites geográficos del rango de los puntos
min_lon, max_lon = event_lon.min() - 1, event_lon.max() + 1 # Calcula los límites de la longitud, añadiendo un margen de 1 grado
min_lat, max_lat = event_lat.min() - 1, event_lat.max() + 1 # Calcula los límites de la latitud, añadiendo un margen de 1 grado

# Normaliza el tiempo transcurrido entre 0 y 1
normalized_time_since_fall = (time_since_fall - time_since_fall.min()) / (time_since_fall.max() - time_since_fall.min())

# Escala el tiempo normalizado al rango de intervalos
scaled_times = np.interp(normalized_time_since_fall, (0, 1), (0, 30))

# Limita el número de eventos de rayos a mostrar
max_events_to_show = 9865

# Selecciona las primeras 'max_events_to_show' filas de las variables relacionadas con los eventos de rayos
event_lat = event_lat[:max_events_to_show]  # Latitud geográfica del evento de rayo
event_lon = event_lon[:max_events_to_show]  # Longitud geográfica del evento de rayo
event_energy = event_energy[:max_events_to_show]  # Energía asociada con el evento de rayo
time_since_fall = time_since_fall[:max_events_to_show]  # Tiempo transcurrido desde la caída
scaled_times = scaled_times[:max_events_to_show]  # Tiempo escalado para la representación en colores

# Ajusta el tiempo a un rango específico
time_range = (0, 30)

# Define los intervalos de la barra de colores manualmente
intervalos = [0, 5, 10, 15, 20, 25, 30]

# Define los intervalos de la barra de colores en incrementos de 5 minutos
intervalos = np.arange(0, 31, 5)

# Personaliza la paleta de colores basada en el tiempo
colores_intervalos = plt.cm.get_cmap('viridis', len(intervalos) - 1) # Obtiene una paleta de colores 'viridis' con un número de colores igual a la cantidad de intervalos - 1
color_map = plt.cm.ScalarMappable(cmap=colores_intervalos) # Crea un objeto ScalarMappable que mapeará los valores de tiempo escalado a colores
color_map.set_array([]) # Establece un arreglo vacío para que el objeto ScalarMappable pueda mapear valores

# Crea un gráfico de dispersión de eventos de rayos con colores de intervalos
fig, ax = plt.subplots(figsize=(12, 8))

# Utiliza make_axes_locatable para dividir el eje principal y añadir un eje de barra de colores
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

# Crea un gráfico de dispersión utilizando la función scatter de Matplotlib
sc = ax.scatter(event_lon, event_lat, c=scaled_times, cmap=colores_intervalos, alpha=0.5, s=10)

# Establece los límites del eje x (longitud) y eje y (latitud) del gráfico principal
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)

# Etiqueta los ejes x e y
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')

# Establece el título del gráfico
ax.set_title('Ubicación de Eventos de Rayos con Tiempo Transcurrido desde la Caída')

# Carga un mapa de fondo usando geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Crea un segundo eje para el minimapa en la esquina superior derecha del mapa principal
minimap_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")

# Limita el minimapa a la región de Cuba
cuba_bounds = [-85, -74, 19, 24]  # Define de los límites geográficos de Cuba: [min_lon, max_lon, min_lat, max_lat]
minimap_ax.set_xlim(cuba_bounds[0], cuba_bounds[1]) # Establece los límites del eje x (longitud) del minimapa para la región de Cuba
minimap_ax.set_ylim(cuba_bounds[2], cuba_bounds[3]) # Establece los límites del eje y (latitud) del minimapa para la región de Cuba

# Dibuja el mapa de Cuba en el minimapa
world_cuba = world.cx[cuba_bounds[0]:cuba_bounds[1], cuba_bounds[2]:cuba_bounds[3]]  # Recorta el mapa mundial a la región de Cuba
world_cuba.boundary.plot(ax=minimap_ax, linewidth=0.8, color='gray')  # Dibuja las fronteras de Cuba en el minimapa con líneas grises
world_cuba.boundary.plot(ax=minimap_ax, linewidth=0.8, color='k')  # Dibuja nuevamente las fronteras de Cuba con líneas negras para resaltar

# Dibuja los eventos de rayos en el minimapa con mayor tamaño y transparencia
minimap_ax.scatter(event_lon, event_lat, c=scaled_times, cmap=colores_intervalos, alpha=0.5, s=25)

# Personaliza el minimapa
minimap_ax.set_xticks([])  # Elimina las marcas del eje x en el minimapa
minimap_ax.set_yticks([])  # Elimina las marcas del eje y en el minimapa

# Restaura los límites originales en el mapa principal
ax.set_xlim(min_lon, max_lon)  # Restaura los límites x en el gráfico principal
ax.set_ylim(min_lat, max_lat)  # Restaura los límites y en el gráfico principal

# Personaliza los colores de las fronteras del mapa y su estilo
world.boundary.plot(ax=ax, linewidth=0.8, color='gray')  # Dibuja las fronteras del mapa mundial en el gráfico principal con líneas grises
world.boundary.plot(ax=ax, linewidth=0.8, color='k')  # Dibuja nuevamente las fronteras del mapa mundial en el gráfico principal con líneas negras para resaltar

# Ajusta la barra de colores para que coincida con los intervalos específicos
cbar = plt.colorbar(color_map, cax=cax, ticks=intervalos, label='Tiempo Transcurrido (minutos)')

# Actualiza el mapeo de colores de la barra con los intervalos específicos
color_map.set_clim(0, 30) # Establece los límites de los colores en la barra de colores

cbar.set_ticklabels([f"{tick} min" for tick in intervalos]) # Configura las etiquetas de los ticks en la barra de colores con intervalos específicos
plt.grid(True) # Muestra la cuadrícula en el gráfico principal
plt.show() # Muestra el gráfico completo con el minimapa y la barra de colores

# Cierra el dataset después de usarlo
ds.close()
####################################################################################################################################################

####################################################################################################################################################
# Importa las bibliotecas necesarias
import xarray as xr  # Para el manejo de datos en formato NetCDF
import matplotlib.pyplot as plt  # Para la creación de gráficos
import geopandas as gpd  # Para trabajar con datos geoespaciales
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Para crear un minimapa en el gráfico
import numpy as np  # Para operaciones numéricas

# Rutas a los archivos NetCDF
file_paths = [
    "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300200_e20231002300400_c20231002300417.nc",
    "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300000_e20231002300200_c20231002300216.nc",
    "/content/drive/MyDrive/Pasantía/OR_GLM-L2-LCFA_G16_s20231002300400_e20231002301000_c20231002301018.nc"
]

# Carga los archivos NetCDF sin especificar la dimensión
datasets = [xr.open_dataset(file_path) for file_path in file_paths]

# Crea un nuevo conjunto de datos combinando todas las variables relevantes
ds_combined = xr.Dataset({
    'event_lat': xr.concat([ds['event_lat'] for ds in datasets], dim='number_of_events'),  # Concatena la variable 'event_lat' de todos los conjuntos de datos a lo largo de la dimensión 'number_of_events'
    'event_lon': xr.concat([ds['event_lon'] for ds in datasets], dim='number_of_events'),  # Concatena la variable 'event_lon' de todos los conjuntos de datos a lo largo de la dimensión 'number_of_events'
    'event_time_offset': xr.concat([ds['event_time_offset'] for ds in datasets], dim='number_of_events'),  # Concatena la variable 'event_time_offset' de todos los conjuntos de datos a lo largo de la dimensión 'number_of_events'
    'event_energy': xr.concat([ds['event_energy'] for ds in datasets], dim='number_of_events'),  # Concatena la variable 'event_energy' de todos los conjuntos de datos a lo largo de la dimensión 'number_of_events'
})


# Accede a las variables relacionadas con la caída de los relámpagos y el tiempo
event_lat = ds_combined['event_lat']  # Latitud geográfica del evento de rayo
event_lon = ds_combined['event_lon']  # Longitud geográfica del evento de rayo
event_time_offset = ds_combined['event_time_offset']  # Tiempo en segundos desde el inicio de la captura del evento de rayo
event_energy = ds_combined['event_energy']  # Energía asociada con el evento de rayo

# Calcula el tiempo real de caída en minutos como valores numéricos
event_time_minutes = event_time_offset.astype(int) / 60.0  # Convertir a tipo int y luego a minutos

# Calcula el tiempo transcurrido desde la caída en minutos
current_time = event_time_minutes.max()  # Tiempo actual (último evento como referencia)
time_since_fall = current_time - event_time_minutes  # Tiempo transcurrido desde la caída

# Obtiene los límites geográficos del rango de los puntos
min_lon, max_lon = event_lon.min() - 1, event_lon.max() + 1  # Establece el límite inferior restando 1 y el límite superior sumando 1 a la longitud geográfica de los eventos de rayos
min_lat, max_lat = event_lat.min() - 1, event_lat.max() + 1  # Establece el límite inferior restando 1 y el límite superior sumando 1 a la latitud geográfica de los eventos de rayos

# Normaliza el tiempo transcurrido entre 0 y 1
normalized_time_since_fall = (time_since_fall - time_since_fall.min()) / (time_since_fall.max() - time_since_fall.min())

# Escala el tiempo normalizado al rango de intervalos
scaled_times = np.interp(normalized_time_since_fall, (0, 1), (0, 30))

# Limita el número de eventos de rayos a mostrar
max_events_to_show = 32862  # Establece el número máximo de eventos de rayos a mostrar
datasets = [ds.isel(number_of_events=slice(max_events_to_show)) for ds in datasets]  # Limita cada conjunto de datos al número máximo de eventos de rayos a mostrar

# Ajusta el tiempo a un rango específico
time_range = (0, 30)

# Define los intervalos de la barra de colores manualmente
intervalos = [0, 5, 10, 15, 20, 25, 30]

# Define los intervalos de la barra de colores en incrementos de 5 minutos
intervalos = np.arange(0, 31, 5)

# Personaliza la paleta de colores basada en el tiempo
colores_intervalos = plt.cm.get_cmap('viridis', len(intervalos) - 1)  # Obtiene una paleta de colores basada en el tiempo (viridis) con la longitud de intervalos menos 1
color_map = plt.cm.ScalarMappable(cmap=colores_intervalos)  # Crea un mapeo de colores escalares utilizando la paleta de colores
color_map.set_array([])  # Establece el arreglo vacío para el mapeo de colores

# Crea un gráfico de dispersión de eventos de rayos con colores de intervalos
fig, ax = plt.subplots(figsize=(12, 8))  # Crea una figura y ejes para el gráfico con un tamaño específico
divider = make_axes_locatable(ax)  # Divide los ejes para agregar una barra de colores
cax = divider.append_axes("right", size="5%", pad=0.1)  # Crea un eje para la barra de colores en el lado derecho del gráfico con un tamaño específico y relleno
sc = ax.scatter(event_lon, event_lat, c=scaled_times, cmap=colores_intervalos, alpha=0.5, s=10)  # Crea un gráfico de dispersión de eventos de rayos con colores de intervalos, opacidad de 0.5 y tamaño de puntos de 10
ax.set_xlim(min_lon, max_lon)  # Establece los límites del eje x (longitud) del gráfico principal
ax.set_ylim(min_lat, max_lat)  # Establece los límites del eje y (latitud) del gráfico principal
ax.set_xlabel('Longitud')  # Etiqueta del eje x
ax.set_ylabel('Latitud')  # Etiqueta del eje y
ax.set_title('Ubicación de Eventos de Rayos con Tiempo Transcurrido desde la Caída')  # Título del gráfico

# Carga un mapa de fondo usando geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Crea un segundo eje para el minimapa en la esquina superior derecha del mapa principal
minimap_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")

# Limita el minimapa a la región de Cuba
cuba_bounds = [-85, -74, 19, 24]  # Define los límites geográficos de la región de Cuba [min_lon, max_lon, min_lat, max_lat]
minimap_ax.set_xlim(cuba_bounds[0], cuba_bounds[1])  # Establece los límites del eje x (longitud) del minimapa para la región de Cuba
minimap_ax.set_ylim(cuba_bounds[2], cuba_bounds[3])  # Establece los límites del eje y (latitud) del minimapa para la región de Cuba

# Dibuja el mapa de Cuba en el minimapa
world_cuba = world.cx[cuba_bounds[0]:cuba_bounds[1], cuba_bounds[2]:cuba_bounds[3]]  # Selecciona la región de Cuba del mapa del mundo
world_cuba.boundary.plot(ax=minimap_ax, linewidth=0.8, color='gray')  # Dibuja las fronteras de Cuba en el minimapa con un grosor de línea de 0.8 y color gris
world_cuba.boundary.plot(ax=minimap_ax, linewidth=0.8, color='k')  # Dibuja las fronteras de Cuba en el minimapa con un grosor de línea de 0.8 y color negro

# Dibuja los eventos de rayos en el minimapa
minimap_ax.scatter(event_lon, event_lat, c=scaled_times, cmap=colores_intervalos, alpha=0.5, s=5)

# Personaliza el minimapa
minimap_ax.set_xticks([])  # Elimina las marcas del eje x en el minimapa
minimap_ax.set_yticks([])  # Elimina las marcas del eje y en el minimapa

# Restaura los límites originales en el mapa principal
ax.set_xlim(min_lon, max_lon)  # Establece los límites del eje x (longitud) del gráfico principal
ax.set_ylim(min_lat, max_lat)  # Establece los límites del eje y (latitud) del gráfico principal

# Personaliza los colores de las fronteras del mapa y su estilo en el gráfico principal
world.boundary.plot(ax=ax, linewidth=0.8, color='gray')  # Dibuja las fronteras del mundo en el gráfico principal con un grosor de línea de 0.8 y color gris
world.boundary.plot(ax=ax, linewidth=0.8, color='k')  # Dibuja las fronteras del mundo en el gráfico principal con un grosor de línea de 0.8 y color negro


# Ajusta la barra de colores para que coincida con los intervalos específicos
cbar = plt.colorbar(color_map, cax=cax, ticks=intervalos, label='Tiempo Transcurrido (minutos)')

# Actualiza el mapeo de colores de la barra con los intervalos específicos
color_map.set_clim(0, 30)  # Establece los límites del mapeo de colores en la barra de colores de 0 a 30 minutos
cbar.set_ticklabels([f"{tick} min" for tick in intervalos])  # Etiqueta las marcas en la barra de colores con intervalos específicos en minutos
plt.grid(True)  # Muestra una cuadrícula en el gráfico principal
plt.show()  # Muestra el gráfico principal con la barra de colores

# Cierra los datasets después de usarlos
for dataset in datasets:
    dataset.close()  # Cierra cada conjunto de datos NetCDF

####################################################################################################################################################

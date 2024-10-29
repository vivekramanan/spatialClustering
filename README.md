# spatialClustering
Geospatial clustering of EJI data for specific states

Written by: Vivek Ramanan
Date: Oct 2024
For use internally for members of the Sarkar Lab at Brown University. 

Contains: 
1. geospatial_clustering.py
2. Data/
3. Figures/

### Dependencies: 
Python3 (I used python 3.9)

Once you've installed python3, make sure the following packages are installed (using pip): 
'''
pip install <package>
'''
1. Pandas
2. Numpy
3. Geopandas
4. Matplotlib
5. Plotly
6. Sklearn
7. json
8. Pysal
9. Splot
10. Esda
11. folium
12. geopy
13. shapely
14. pygeoda
15. networkx
16. markov_clustering
17. scipy 

## Geospatial_clustering.py 

This script runs two methods for unsupervised clustering using geospatial attributes, EJI data, and a zipcode-census tract conversion dataset. It also includes feature importance for each clustering method using a Random Forest machine learning methodology. Finally, choropleth plots are used to visualize the clusters using preexisting datafiles saved in Data. 
The two methods of choice: (1) Local Morans and (2) Markov Clustering (also called MCL). 

### INPUTS: 
1. ejiData.csv
2. TRACT_ZIP_122023.csv
3. STATE NAME (for example: RI)

### EXAMPLE RUN: 
python3 geospatial_clustering.py Data/ejiData.csv Data/TRACT_ZIP_122023.csv MA

This would run geospatial clustering on Massachusetts (MA).

## Data/

The data folder contains 3 main files / folders: 
1. ejiData.csv: The original EJI Data download from the CDC
2. TRACT_ZIP_122023.csv: census tract to zipcode conversion (most recent download is 12/20/2023, update accordingly)
3. State-zip-code-GeoJSON: geoJSON files for state zipcode data for plotly choropleth plots

## Figures/

There are a couple figures created from the geospatial clustering python script. Example figures are present for a run of Rhode Island: 
1. <state>_spatialWeights.png: network graph created from the geospatial method
2. <state>_localmorans.png: Morans scatter plot for the cluster structures (check this to explain moran map)
3. <state>_moranMap.png: MAP version of the Local Morans clusterings
4. <state>_Morans_featureImp.png: feature importance of the Morans clustering
5. <state>_network.png: network version of the state for markov clustering
6. <state>_mclMap.png: Markov clustering MAP version
7. <state>_MCL_featureImp.png: feature importance of the markov clustering

import pandas as pd
import numpy as np
import collections
import os
import glob
import math
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import squareform
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI, Query
from FMTSClust import *
import json

app = FastAPI()


# FMTSClustAPI - Step 4 - CLUSTERING ENDPOINTS

### K-means
@app.get('/sil_elbow_scores_TSKMeans')  
async def sil_elbow_scores_TSKMeans(dfName):
    
    if (dfName == "fawn"):
        
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        
        sil_scores, elbow_plot_distances = sil_and_elbow_scores_TSKMeans(df_array_std,11,"euclidean",10,150)

        return {"sil_scores": sil_scores, "wcss": elbow_plot_distances}
        
    elif (dfName == "simagro"):
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})
        
        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        
        sil_scores, elbow_plot_distances = sil_and_elbow_scores_TSKMeans(df_array_std,11,"euclidean",10,150)

        return {"sil_scores": sil_scores, "wcss": elbow_plot_distances}
        
    else:
        return "Invalid database"
    

@app.get('/cluster_TSKmeans')  
async def cluster_TSKmeans(dfName,nClusters):
    
    if (dfName == "fawn"):
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)
        
        
        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)
        
        unique_station_ids = df_std['StationID'].unique()
        
        y_pred_Kmeans_std = cluster_labels_TSKmeans(df_array_std,int(nClusters),"euclidean",10,50)
        
        rmse = calculate_cluster_rmse(flattened_df_std, y_pred_Kmeans_std)
        
        
        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in y_pred_Kmeans_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
        
    elif (dfName == "simagro"):
        
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})
        
        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)
            
        unique_station_ids = df_std['StationID'].unique()
        
        y_pred_Kmeans_std = cluster_labels_TSKmeans(df_array_std,int(nClusters),"euclidean",10,50)
        
        rmse = calculate_cluster_rmse(flattened_df_std, y_pred_Kmeans_std)
        
        
        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in y_pred_Kmeans_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
    
    else:
        return "Invalid database"
    
    

### K-medoids
@app.get('/sil_elbow_scores_TSKMedoids')  
def sil_elbow_scores_TSKMedoids(dfName):
    
    if (dfName == "fawn"):
        
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)

        sil_scores, elbow_plot_distances = sil_and_elbow_scores_TSKMedoids(df_array_std, 11, 300)

        return {"sil_scores": sil_scores, "wcss": elbow_plot_distances}
    
    elif (dfName == "simagro"):
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})

        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)

        sil_scores, elbow_plot_distances = sil_and_elbow_scores_TSKMedoids(df_array_std, 11, 300)

        return {"sil_scores": sil_scores, "wcss": elbow_plot_distances}
    
    else:
        return "Invalid database"

@app.get('/cluster_TSKMedoids')  
def cluster_TSKMedoids(dfName,nClusters):
    
    if (dfName == "fawn"):
        
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)
        
        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)
            
        unique_station_ids = df_std['StationID'].unique()
        
        y_pred_Kmedoids_std, cluster_medoids_std = cluster_labels_TSKMedoids(df_array_std, int(nClusters), 300)
        
        rmse = calculate_cluster_rmse(flattened_df_std, y_pred_Kmedoids_std)
            
        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in y_pred_Kmedoids_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
        
    elif (dfName == "simagro"):
        
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})
        
        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)

        unique_station_ids = df_std['StationID'].unique()
        
        y_pred_Kmedoids_std, cluster_medoids_std = cluster_labels_TSKMedoids(df_array_std, int(nClusters), 300)
        
        rmse = calculate_cluster_rmse(flattened_df_std, y_pred_Kmedoids_std)
            
        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in y_pred_Kmedoids_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
    
    else:
        return "Invalid database"
    
    
### Hierarchical Agglomerative Clustering
@app.get('/sil_score_TSHClustering')  
def sil_score_TSHClustering(dfName):
    
    if (dfName == "fawn"):
        
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)

        return sil_score_TSHierarcClustering(df_array_std)
    
        
    elif (dfName == "simagro"):
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})
        
        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        
        return sil_score_TSHierarcClustering(df_array_std)
    
    else:
        return "Invalid database"
    

@app.get('/cluster_TSHClustering')      
def cluster_TSHClustering(dfName,nClusters):

    if (dfName == "fawn"):
        
        path = "./Data_FAWN"
        df = getFAWNdata(path)

        # Preprocessing
        df_filtered = filterByDate(df, "2003", "2023", "10-31", "06-01")
        df_stations, stations_ids = check_data_availability_stations(df_filtered, 20)
        QC_missing_null_values(df_stations,"fawn")
        QC_outliers(df_stations)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_stations,columns)
        df_std = standardization(df_days, columns)
        
        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 151 = May 31
        doy_lower = 151

        # DOY = 305 = November 1
        doy_upper = 305

        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)
        
        unique_station_ids = df_std['StationID'].unique()
        
        h_cluster_labels_std = TSHierarcClustering(df_array_std, int(nClusters))
        
        rmse = calculate_cluster_rmse(flattened_df_std, h_cluster_labels_std)        

        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in h_cluster_labels_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
        
    elif (dfName == "simagro"):
        
        path = "./Data_SIMAGRO"
        df = getSIMAGROdata(path)
        df = df.rename(columns={"Temp": "Temperature", "RelHum": "RelatHumidity", "Rain":"Rainfall"})
        
        # Preprocessing
        df_filtered = filterByDate(df, "2020", "2023", "09-30", "04-01")
        QC_missing_null_values(df_filtered,"simagro")
        QC_outliers(df_filtered)

        columns=['Temperature','RelatHumidity','Rainfall','SolarRad']
        df_days = daily_aggregation(df_filtered,columns)
        df_std = standardization(df_days, columns)

        allColumns=['Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']

        # DOY = 90 = March 31
        doy_lower = 90
        # DOY = 274 = October 1
        doy_upper = 274
        
        df_array_std = transform_dfTS_to_3Darray_DOYsorted(df_std,allColumns,doy_lower,doy_upper)
        flattened_df_std = flatten_multivariate_time_series(df_array_std)
        
        unique_station_ids = df_std['StationID'].unique()

        h_cluster_labels_std = TSHierarcClustering(df_array_std, int(nClusters))
        
        rmse = calculate_cluster_rmse(flattened_df_std, h_cluster_labels_std)
        
        str_keys = []
        for i in unique_station_ids:
            str_keys.append(""+str(i))
        
        str_values = []
        for i in h_cluster_labels_std:
            str_values.append(int(i))
            
        result = {str_keys[i]: str_values[i] for i in range(len(str_keys))}
        
        return {"clustering": result, "rmse": rmse}
    
    else:
        return "Invalid database"

    

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

color_temp='#030303' #black
color_relHum='#66CD00' #green
color_rainfall='#1874CD' #blue
color_solarRad='#CD6600' #orange

color_C0='#D35400' # orange
color_C1='#34495E' # dark blue
color_C2='#45B39D' # light green
color_C3='#F4D03F' # yellow
color_C4='#808080' # gray
color_C5='#8A2BE2' # blue violet
color_C6='#008000' # green
color_Cblack ='#000000' # black

# FAWN Map coordinates
BBoxFAWN = ((-87.649, -79.728, #longitude
          23.996, 31.062)) #latitude

# SIMAGRO Map coordinates
BBoxSIMAGRO = ((-57.711, -49.449, #longitude
          -33.934, -27.059)) #latitude

def getFAWNdata(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    csv_files.sort()

    # Parse date formats to Date
    big_df = pd.DataFrame()
    dfs = []

    for file in csv_files:
        
        df = pd.read_csv(file)
        
        # Get columns: 0 (Station ID), 1 (date), 8 (avg temperature 2m), 14 (avg relative humidity 2m), 
        # 18 (rain 2m sum), 23 (solar radiation 2m sum)
        dfFiltered = df.iloc[:,[0,1,8,14,18,23]]
        dfFiltered.columns = ['StationID', 'Date_raw', 'Temperature', 'RelatHumidity', 'Rainfall', 'SolarRad']
        dfFiltered.insert(loc = 2, column = 'Date', value = "")
    
    
        # ***********************************
        # 2022 to 2023
        # Date format: YYYY-MM-DD HH:MM:SS
        if ( ("2022" in file) or ("2023" in file) ):
            for index, row in dfFiltered.iterrows():
                dateStr = datetime.strptime(row['Date_raw'], '%Y-%m-%d %H:%M:%S')
                dfFiltered.at[index,'Date'] = dateStr.date()
    
        # ***********************************
        # 1997 to 2021
        # Date format: YYYY-MM-DD
        else:
            for index, row in dfFiltered.iterrows():
                dateStr = datetime.strptime(row['Date_raw'], '%Y-%m-%d')
                dfFiltered.at[index,'Date'] = dateStr.date()
            
    
        dfAppend = dfFiltered[["StationID","Date", "Temperature", "RelatHumidity", "Rainfall", "SolarRad"]]
        dfs.append(dfAppend)
    

    big_df = pd.concat(dfs, ignore_index=True)


    # Drop Feb-29 of leap years: 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024
    # to map "Day of Year" (DOY) between 1 and 365
    big_df['Date'] = pd.to_datetime(big_df['Date'])
    big_df = big_df.drop(big_df[big_df['Date']=="1996-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2000-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2004-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2008-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2012-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2016-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2020-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2024-02-29"].index)

    
    # DOY: from date (YYYY-MM-DD) to day of year (doy = 1-366)
    big_df['doy']=0

    for index, row in big_df.iterrows():
        objDate = row['Date']
        day = objDate.timetuple().tm_yday
        year = objDate.timetuple().tm_year
        mon = objDate.timetuple().tm_mon
    
        #2000  2004  2008  2012  2016  2020
        if (year == 2000 or year == 2004 or year == 2008 or year == 2012 or year == 2016 or year == 2020):
            if (mon > 2):
                day = day - 1
    
        big_df.at[index,'doy'] = day

    return big_df



def getSIMAGROdata(path):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    csv_files.sort()

    big_df = pd.DataFrame()
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df["Date"], format='%d/%m/%y')
        dfs.append(df)

    big_df = pd.concat(dfs, ignore_index=True)
    
    
    
    
    # Drop Feb-29 of leap years: 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024
    # to map "Day of Year" (DOY) between 1 and 365
    big_df['Date'] = pd.to_datetime(big_df['Date'])
    big_df = big_df.drop(big_df[big_df['Date']=="1996-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2000-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2004-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2008-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2012-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2016-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2020-02-29"].index)
    big_df = big_df.drop(big_df[big_df['Date']=="2024-02-29"].index)
    
    
    big_df['Temp'] = pd.to_numeric(big_df['Temp'], errors='coerce')
    big_df['RelHum'] = pd.to_numeric(big_df['RelHum'], errors='coerce')
    big_df['Rain'] = pd.to_numeric(big_df['Rain'], errors='coerce')
    big_df['SolarRad'] = pd.to_numeric(big_df['SolarRad'], errors='coerce')
    #big_df['Date'] = pd.to_datetime(df["Date"], format='%d/%m/%y')
    #big_df['Date'] = pd.to_datetime(df["Date"])
    
    # daily values
    big_df = big_df.groupby(['StationID','Date']).agg({'Temp':'mean', 'RelHum':'mean', 'Rain':'sum', 'SolarRad':'sum'})
    big_df = big_df.reset_index()

    big_df['SolarRad'] = big_df['SolarRad'].div(1000).round(2)
    
    # DOY: from date (YYYY-MM-DD) to day of year (doy = 1-366)
    big_df['doy']=0

    for index, row in big_df.iterrows():
        objDate = row['Date']
        day = objDate.timetuple().tm_yday
        year = objDate.timetuple().tm_year
        mon = objDate.timetuple().tm_mon
    
        #2000  2004  2008  2012  2016  2020
        if (year == 2000 or year == 2004 or year == 2008 or year == 2012 or year == 2016 or year == 2020):
            if (mon > 2):
                day = day - 1
    
        big_df.at[index,'doy'] = day

    return big_df
    

# Filtering data by season and years
def filterByDate(df, yearStart, yearEnd, monthDayStart, monthDayEnd):
    
    # delete data before yearStart
    firstDate = yearStart+"-01-01"
    df = df[~(df['Date'] < firstDate)]
    
    # delete data after yearEnd
    lastDate = yearEnd+"-12-31"
    df = df[~(df['Date'] < firstDate)]


    # delete between seasons, every year
    for i in range(int(yearStart),(int(yearEnd)+1)):
        from_ts = str(i)+"-"+monthDayEnd
        to_ts = str(i)+"-"+monthDayStart
        df = df[(df['Date'] < from_ts) | (df['Date'] > to_ts)]


    #delete the start of the first year
    to_ts = yearStart+"-"+monthDayStart
    df = df[(df['Date'] > to_ts)]

    #delete the end of the last yer
    from_ts = yearEnd+"-"+monthDayEnd
    df = df[(df['Date'] < from_ts)]

    return df


def check_data_availability_stations(df, minYears):
    ids = df['StationID'].unique()
    
    # Delete data from unkown stations

    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]

    realStations=[]
    unknownStations=[]
    
    for ID in ids:
        if ID in names['StationID'].values:
            realStations.append(ID)
        else:
            unknownStations.append(ID)
        
    # clean data from unkownd stations
    for station in unknownStations:
        df.drop(df[(df['StationID'] == station)].index, inplace=True)
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # check if stations have data for all years
    df_byStationYearMonth = df.groupby([df['StationID'], df['Date'].dt.year.rename('year'),df['Date'].dt.month.rename('month')]).agg({'count'})
    df_byStationYearMonth = df_byStationYearMonth.reset_index()
    df_byStationYear = df_byStationYearMonth[['StationID','year','month']]
    


    stationYears = {}

    for station in realStations:
        df_station = df_byStationYear.loc[df_byStationYear['StationID'] == station]
        df_aux = df_station['year'].unique()
        stationYears[station] = np.sort(df_aux)
    
    ordStationYears = collections.OrderedDict(sorted(stationYears.items()))

    # filter data from stations with few data (less equal than 20 years)
    for station in ordStationYears:
        nYears = len(ordStationYears[station])
        if nYears <= minYears:
            df.drop(df[(df['StationID'] == station)].index, inplace=True)

    ids_filtered = df['StationID'].unique()

    return df, ids_filtered
    

def QC_missing_null_values(df,dfName):

    # Maximum of measurements: 
    # t = measurements per day
    # d = days per season
    # y = number of years/seasons
    # s = number of stations
    # MaxMeas = t * d * y * s
        
    maxMeasurements = 0
    
    if (dfName == "fawn"):
        maxMeasurements = 1*212*20*29
    elif (dfName == "simagro"):
        maxMeasurements = 1*182*3*19
    else:
        print("Invalid dataset.")
        return
    

    realMeasurements = len(df.index)

    missingValues = maxMeasurements - realMeasurements

    print("Maximum number of measurements: "+str(maxMeasurements))
    print("Real number of measurements: "+str(realMeasurements))
    print("Missing measurements: "+str(missingValues))
    
    print("Missing Values (%): "+str(missingValues/maxMeasurements))
    
    
    null_count = df.isnull().sum().sum()
    print("\nNull values: "+str(null_count))
    print("%: "+str(null_count/maxMeasurements))
    
    



def QC_outliers(df):
    
    factor = 1.5
    
    
    # Temperature
    q1_T = df["Temperature"].quantile(0.25)
    q3_T = df["Temperature"].quantile(0.75)
        
    iqr_T = q3_T - q1_T
        
    upper_limit_T = q3_T + (factor*iqr_T)
    lower_limit_T = q1_T - (factor*iqr_T)

    T_outliers = np.where(df["Temperature"] > upper_limit_T, True,
             np.where(df["Temperature"] < lower_limit_T, True, False))

    T_without_outliers = df.loc[~(T_outliers)]

    print("Upper_limit_T: "+str(upper_limit_T))
    print("Lower_limit_T: "+str(lower_limit_T))

    print("Temperature Outliers: "+str(df.shape[0]-T_without_outliers.shape[0]))

    # capping outliers with upper and lower limits
    df["Temperature"] = np.where(df["Temperature"]>upper_limit_T, upper_limit_T,
                 np.where(df["Temperature"]<lower_limit_T, lower_limit_T, df["Temperature"]))
    
    
    # Relative Humidity
    q1_RH = df["RelatHumidity"].quantile(0.25)
    q3_RH = df["RelatHumidity"].quantile(0.75)

    iqr_RH = q3_RH - q1_RH

    upper_limit_RH = q3_RH + (factor*iqr_RH)
    lower_limit_RH = q1_RH - (factor*iqr_RH)

    RH_outliers = np.where(df["RelatHumidity"] > upper_limit_RH, True,
                 np.where(df["RelatHumidity"] < lower_limit_RH, True, False))

    RH_without_outliers = df.loc[~(RH_outliers)]

    print("Upper_limit_RH: "+str(upper_limit_RH))
    print("Lower_limit_RH: "+str(lower_limit_RH))

    print("Relative Humidity Outliers: "+str(df.shape[0]-RH_without_outliers.shape[0]))

    # capping outliers with upper and lower limits
    df["RelatHumidity"] = np.where(df["RelatHumidity"]>upper_limit_RH, upper_limit_RH,
                     np.where(df["RelatHumidity"]<lower_limit_RH, lower_limit_RH, df["RelatHumidity"]))



    # Rainfall
    q1_RF = df["Rainfall"].quantile(0.25)
    q3_RF = df["Rainfall"].quantile(0.75)

    iqr_RF = q3_RF - q1_RF

    upper_limit_RF = q3_RF + (factor*iqr_RF)
    lower_limit_RF = q1_RF - (factor*iqr_RF)

    RF_outliers = np.where(df["Rainfall"] > upper_limit_RF, True,
                 np.where(df["Rainfall"] < lower_limit_RF, True, False))

    RF_without_outliers = df.loc[~(RF_outliers)]

    print("Upper_limit_RF: "+str(upper_limit_RF))
    print("Lower_limit_RF: "+str(lower_limit_RF))

    print("Rainfall Outliers: "+str(df.shape[0]-RF_without_outliers.shape[0]))

    # capping outliers with upper and lower limits
    df["Rainfall"] = np.where(df["Rainfall"]>upper_limit_RF, upper_limit_RF,
                     np.where(df["Rainfall"]<lower_limit_RF, lower_limit_RF, df["Rainfall"]))

    
    
    # Solar Radiation
    q1_SR = df["SolarRad"].quantile(0.25)
    q3_SR = df["SolarRad"].quantile(0.75)

    iqr_SR = q3_SR - q1_SR

    upper_limit_SR = q3_SR + (factor*iqr_SR)
    lower_limit_SR = q1_SR - (factor*iqr_SR)

    SR_outliers = np.where(df["SolarRad"] > upper_limit_SR, True,
                 np.where(df["SolarRad"] < lower_limit_SR, True, False))

    SR_without_outliers = df.loc[~(SR_outliers)]

    print("Upper_limit_SR: "+str(upper_limit_SR))
    print("Lower_limit_SR: "+str(lower_limit_SR))

    print("Solar Radiation Outliers: "+str(df.shape[0]-SR_without_outliers.shape[0]))

    # capping outliers with upper and lower limits
    df["SolarRad"] = np.where(df["SolarRad"]>upper_limit_SR, upper_limit_SR,
                     np.where(df["SolarRad"]<lower_limit_SR, lower_limit_SR, df["SolarRad"]))
    

    

def daily_aggregation(df,agg_columns):

    df_days = df.groupby(['StationID','doy'])[agg_columns].mean()
    df_days = df_days.reset_index()

    
    print("Minimum and maximum values for each variable:")
    print("> Temperature:")
    print("Min: "+str(round(df_days['Temperature'].min(),2))+
          "ºC     Max: "+str(round(df_days['Temperature'].max(),2))+"ºC\n")
    print("> Relative Humidity:")
    print("Min: "+str(round(df_days['RelatHumidity'].min(),2))+
          "%      Max: "+str(round(df_days['RelatHumidity'].max(),2))+"%\n")
    print("> Rainfall:")
    print("Min: "+str(round(df_days['Rainfall'].min(),2))+
          "\"       Max: "+str(round(df_days['Rainfall'].max(),2))+"\"\n")
    print("> Solar Radiation:")
    print("Min: "+str(round(df_days['SolarRad'].min(),2))+
          " MJm2  Max: "+str(round(df_days['SolarRad'].max(),2))+" MJm2\n")
    
    return df_days


def standardization(df, std_columns):
    
    data_to_standardize = df[std_columns]
    
    scaler = StandardScaler().fit(data_to_standardize)

    standardized_data = scaler.transform(data_to_standardize)

    df_std = pd.DataFrame(standardized_data, columns = std_columns)

    df_std.insert(loc=0, column='StationID', value=df['StationID'])
    df_std.insert(loc=1, column='doy', value=df['doy'])
    
    return df_std
    
    
    
def transform_dfTS_to_3Darray_DOYsorted(df, columns, doy_lim1, doy_lim2):
    
    # Get the unique values of StationID
    unique_station_ids = df['StationID'].unique()
    
    # Create an empty list to hold the dataframes
    split_dfs = []

    # Loop over the unique values of StationID and split the dataset accordingly
    for station_id in unique_station_ids:
        df_temp = df[df['StationID'] == station_id]
        firstMonthsYear = df_temp[df_temp["doy"] <= doy_lim1] # until 
        lastMonthsYear = df_temp[df_temp["doy"] >= doy_lim2] # from 
        df_season = pd.concat([lastMonthsYear, firstMonthsYear])
        split_dfs.append(df_season[columns])
    
    
    # Get the length of the first two dimensions of the array
    dim1 = len(split_dfs)
    dim2 = split_dfs[0].shape[0]
    dim3 = split_dfs[0].shape[1]


    df_return = np.zeros((dim1, dim2, dim3))

    # Copy the values from the dataframes to the new array
    for i, data in enumerate(split_dfs):
        df_return[i, :, :] = data.values
    
    return df_return




def plot_first_TS(array, station_list):
    
    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]
    
    index = 0
    
    plt.figure(figsize=(20,3))
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    plt.title("Station "+str(stationId)+" - "+stationName+" - (Temperature, Rel. Humidity, Rainfall, Solar Radiation)")
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    

    plt.plot(var_temp, '-',color=color_temp)
    plt.plot(var_relHum, '-',color=color_relHum)
    plt.plot(var_rain, '-',color=color_rainfall)
    plt.plot(var_solRad, '-',color=color_solarRad)
    
    #for index, item in enumerate(array):
    #    plt.figure(figsize=(20,3))
    #    stationId = station_list[index]
    #    result = names[names['StationID'] == stationId]
    #    stationName = result.iloc[0]['StationName']
    #    plt.title(str(stationId)+" - "+stationName)
    #
    #
    #    for i in item:
    #        plt.plot(item, '-',c=(np.random.random(), np.random.random(), np.random.random()))
    #
    #    break

        
def plot_all_TS(array, station_list):
    
    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]
    
    for index, item in enumerate(array):
        plt.figure(figsize=(20,3))
        stationId = station_list[index]
        result = names[names['StationID'] == stationId]
        stationName = result.iloc[0]['StationName']
        plt.title(str(stationId)+" - "+stationName)
    
    
        for i in item:
            plt.plot(item, '-',c=(np.random.random(), np.random.random(), np.random.random()))
    

def plot_TS_by_index(array, station_list, index):
    
    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]
    
    
    plt.figure(figsize=(10,3))
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    plt.title("Station "+str(stationId)+" - "+stationName)#+" - T (black), RH (green), RF (blue), SR (orange)")#+" - Temperature (black), Relative Humidity (green), Rainfall (blue), Solar Radiation (orange)")
    
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    
    plt.xlabel("Days")
    plt.margins(x=0.02, y=0.25)
    plt.plot(var_temp, '-',color=color_temp,label='T')
    plt.plot(var_relHum, '-',color=color_relHum,label='RH')
    plt.plot(var_rain, '-',color=color_rainfall,label='RF')
    plt.plot(var_solRad, '-',color=color_solarRad,label='SR')
    plt.legend(loc='upper center', frameon=False, ncol=4)

def plot_TS_by_index_title(array, station_list, index, title):
    
    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]
    
    plt.figure(figsize=(10,3))
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    station_name_temp = stationName.rsplit(' (')[0]
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" --- "+title)#+" - Temperature (black), Relative Humidity (green), Rainfall (blue), Solar Radiation (orange)")
    
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item: # 182 itens
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    
    plt.xlabel("Days")
    plt.margins(x=0.02, y=0.25)
    plt.plot(var_temp, '-',color=color_temp,label='T')
    plt.plot(var_relHum, '-',color=color_relHum,label='RH')
    plt.plot(var_rain, '-',color=color_rainfall,label='RF')
    plt.plot(var_solRad, '-',color=color_solarRad, label='SR')
    plt.legend(loc='upper center', frameon=False, ncol=4)


    
def plot_TS_by_index_separated_variables(array, station_list, index):
    
    stations = pd.read_csv('StationsList.csv')
    names = stations[['StationName','StationID']]
    
    stationId = station_list[index]
    result = names[names['StationID'] == stationId]
    stationName = result.iloc[0]['StationName']
    
    var_temp = []
    var_relHum = []
    var_rain = []
    var_solRad = []
    
    item = array[index]

    for i in item:
        var_temp.append(i[0])
        var_relHum.append(i[1])
        var_rain.append(i[2])    
        var_solRad.append(i[3])
    
    station_name_temp = stationName.rsplit(' (')[0]
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Temperature")
    plt.xlabel("Days")
    plt.ylabel("°C")
    plt.margins(x=0.01)
    plt.plot(var_temp, '-',color=color_temp)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Relative Humidity")
    plt.xlabel("Days")
    plt.ylabel("%")
    plt.margins(x=0.01)
    plt.plot(var_relHum, '-',color=color_relHum)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Rainfall")
    plt.xlabel("Days")
    plt.ylabel("in")
    plt.margins(x=0.01)
    plt.plot(var_rain, '-',color=color_rainfall)
    
    plt.figure(figsize=(10,3))
    plt.title("Station "+str(stationId)+" - "+station_name_temp+" - Solar Radiation")
    plt.xlabel("Days")
    plt.ylabel("MJ/m²")
    plt.margins(x=0.01)
    plt.plot(var_solRad, '-',color=color_solarRad)

    

def sil_and_elbow_scores_TSKMeans(df_array, max_cluster, metric_c, n_init_c, max_iter_c):
    
    sil_scores = []
    elbow_plot_distances = []

    K = range(1,max_cluster)

    for k in K:
        # model instantiation
        model = TimeSeriesKMeans(n_clusters=k, metric=metric_c, n_init=n_init_c, max_iter=max_iter_c)
        y_pred = model.fit_predict(df_array)
    
        # silhouette score
        if (k>1):
            score=silhouette_score(df_array, y_pred, metric=metric_c)
            sil_scores.append(score)
    
        # inertia, for elbow plot
        elbow_plot_distances.append(model.inertia_)

    c=2
    for i in sil_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, elbow_plot_distances, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Plot - K-means')
    plt.show()
    
    
def cluster_labels_TSKmeans(df_array, n_clusters_c, metric_c, n_init_c, max_iter_c):
    model = TimeSeriesKMeans(n_clusters=n_clusters_c, metric=metric_c, n_init=n_init_c, max_iter=max_iter_c)
    y_pred = model.fit_predict(df_array)
    return y_pred


def sil_score_TSHierarcClustering(df_array):
    
    # Calculate the DTW distance matrix
    positions = df_array.shape[0]
    dtw_distance_matrix = np.zeros((positions, positions))
    for i in range(positions):
        for j in range(positions):
            dtw_distance_matrix[i, j] = dtw(df_array[i], df_array[j])
            
    
    # Convert the distance matrix to condensed form
    dtw_distance_condensed = squareform(dtw_distance_matrix)

    # Perform hierarchical clustering on the condensed DTW distance matrix
    linkage_matrix = linkage(dtw_distance_condensed, method='ward')

    sil_scores = []
 
    for i in range(2,dtw_distance_matrix.shape[0]):
    
        n_clusters = i # Change this to the desired number of clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        score=silhouette_score(dtw_distance_matrix, cluster_labels, metric="precomputed")
        sil_scores.append(score)

    c=2
    for i in sil_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1

        
def dendogram_TSHierarcClustering(df_array, num_clusters, list_names_dendogram):
    
    # Calculate the DTW distance matrix
    positions = df_array.shape[0]
    dtw_distance_matrix = np.zeros((positions, positions))
    for i in range(positions):
        for j in range(positions):
            dtw_distance_matrix[i, j] = dtw(df_array[i], df_array[j])

            
    # Convert the distance matrix to condensed form
    dtw_distance_condensed = squareform(dtw_distance_matrix)

    # Perform hierarchical clustering on the condensed DTW distance matrix
    linkage_matrix = linkage(dtw_distance_condensed, method='ward')

    # Perform clustering by cutting the dendrogram at a suitable height
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    #print(cluster_labels)
    
    plt.figure(figsize=(13, 10))
    dendrogram(
            linkage_matrix,
            orientation='right',
            labels=list_names_dendogram,
            distance_sort='descending',
            show_leaf_counts=False
          )
    plt.show()

    return cluster_labels, linkage_matrix

def plot_map_stations(stations,dfName):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    if (dfName == "fawn"):
        BBox = BBoxFAWN
    elif (dfName == "simagro"):
        BBox = BBoxSIMAGRO
    else:
        print("Invalid dataset.")
        return
    
    BBox = ((-87.649, -79.728, #longitude
          23.996, 31.062)) #latitude
    
    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))


    ax.scatter(df_plot.longitude, df_plot.latitude, zorder=1, alpha=1, c="black", s=40)
    
    ax.set_title('Meteorological Stations - Florida/US')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    for index, row in df_plot.iterrows():
        plt.text(row['longitude']+0.2,row['latitude']+0.1, row['StationName'], horizontalalignment='center', size='medium', color='black')

        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')

    
def plot_map_stations_IDs(stations, dfName):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    if (dfName == "fawn"):
        BBox = BBoxFAWN
    elif (dfName == "simagro"):
        BBox = BBoxSIMAGRO
    else:
        print("Invalid dataset.")
        return
    
    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))


    ax.scatter(df_plot.longitude, df_plot.latitude, zorder=1, alpha=1, c="black", s=40)
    
    ax.set_title('Meteorological Stations - Florida/US')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    plt.rcParams.update({'font.size':9})
        
    for index, row in df_plot.iterrows():
        #st_name = row['StationName'].rsplit(' (')[0]
        st_id = row['StationID']
        if ((st_id == 290) | (st_id == 320) | (st_id == 302)):
            plt.text(row['longitude']+0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')            
        else: 
            plt.text(row['longitude'],row['latitude']-0.22, st_id, horizontalalignment='center', size='medium', color='black')

        
        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')
        
    
    
    
def plot_map_clustering(stations, cluster_labels, dfName):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    if (dfName == "fawn"):
        BBox = BBoxFAWN
    elif (dfName == "simagro"):
        BBox = BBoxSIMAGRO
    else:
        print("Invalid dataset.")
        return
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))

    
    for i in range(0,len(n_clusters)):
        if (i==0):
            color = color_C0
        else:
            color = color_C1
        ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title('Weather Stations clusters on Florida Map')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    for index, row in df_plot.iterrows():
        plt.text(row['longitude']+0.2,row['latitude']+0.1, row['StationName'], horizontalalignment='center', size='medium', color='black')

        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')

    
def plot_map_clustering_names(stations, cluster_labels, title, dfName):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    if (dfName == "fawn"):
        BBox = BBoxFAWN
    elif (dfName == "simagro"):
        BBox = BBoxSIMAGRO
    else:
        print("Invalid dataset.")
        return
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))

    for i in range(0,len(n_clusters)):
        if (len(n_clusters)==1):
            color = color_Cblack
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)
        else:
            if (i==0):
                color = color_C0
            elif (i==1):
                color = color_C1
            elif (i==2):
                color = color_C2
            elif (i==3):
                color = color_C3
            elif (i==4):
                color = color_C4
            elif (i==5):
                color = color_C5    
            else:
                color = color_C6
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title(title)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    if (dfName == "fawn"):
        for index, row in df_plot.iterrows():
            st_name = row['StationName'].rsplit(' (')[0]
            plt.text(row['longitude']-0.1,row['latitude']+0.1, st_name, 
                     horizontalalignment='center', size='medium',    color='black')

    elif (dfName == "simagro"):
        for index, row in df_plot.iterrows():
            st_name = row['StationName'].rsplit(' (')[0]
            if (st_name == "Itaqui") | (st_name=="Cachoeira do Sul") | (st_name=="Piratini"):
                plt.text(row['longitude'],row['latitude']+0.1, st_name, 
                         horizontalalignment='center', size='medium', color='black')            
            else: 
                plt.text(row['longitude']+0.1,row['latitude']-0.25, st_name, 
                     horizontalalignment='center', size='medium', color='black') 
        
        
        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')
        

def plot_map_clustering_IDs(stations, cluster_labels, title, dfName):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    if (dfName == "fawn"):
        BBox = BBoxFAWN
    elif (dfName == "simagro"):
        BBox = BBoxSIMAGRO
    else:
        print("Invalid dataset.")
        return
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))

    
    

        
    for i in range(0,len(n_clusters)):
        if (len(n_clusters)==1):
            color = color_Cblack
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)
        else:
            if (i==0):
                color = color_C0
            elif (i==1):
                color = color_C1
            elif (i==2):
                color = color_C2
            elif (i==3):
                color = color_C3
            elif (i==4):
                color = color_C4
            elif (i==5):
                color = color_C5    
            else:
                color = color_C6
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title(title)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    plt.rcParams.update({'font.size':9})
    
    for index, row in df_plot.iterrows():
        #st_name = row['StationName'].rsplit(' (')[0]
        st_id = row['StationID']
        if ((st_id == 290) | (st_id == 320) | (st_id == 302)):
            plt.text(row['longitude']+0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        elif (st_id == 360):
            plt.text(row['longitude']-0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        else: 
            plt.text(row['longitude'],row['latitude']-0.22, st_id, horizontalalignment='center', size='medium', color='black')
            
        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')

    

    
    
def cluster_labels_TSKMedoids(df_array, n_clusters_c, max_iter_c):
    
    # Reshape the data to (n_samples, n_timestamps, n_features)
    n_samples, n_timestamps, n_features = df_array.shape
    X = df_array.reshape((n_samples, n_timestamps, n_features))

    # Perform k-medoids clustering with "euclidean" as the distance metric
    kmedoids = KMedoids(n_clusters=n_clusters_c, metric="euclidean", random_state=0, max_iter=max_iter_c)
    y_pred = kmedoids.fit_predict(X.reshape(n_samples, -1))

    # To access cluster medoids, reshape them back to the original format
    cluster_medoids_indices = kmedoids.medoid_indices_
    cluster_medoids = X[cluster_medoids_indices]
    
    return y_pred, cluster_medoids


def sil_and_elbow_scores_TSKMedoids(df_array, max_cluster, max_iter_c):
    
    
    n_samples, n_timestamps, n_features = df_array.shape
    X = df_array.reshape((n_samples, n_timestamps * n_features))

    # Range of cluster numbers to try
    K = range(1,max_cluster)

    # wcss for elbow plot
    elbow_plot_distances = []
    
    # Calculate silhouette scores for each k
    silhouette_scores = []
    
    for k in K:
        kmedoids = KMedoids(n_clusters=k, metric="euclidean", random_state=0, max_iter=max_iter_c)
        y_pred = kmedoids.fit_predict(X)

        if (k>1):
            silhouette_scores.append(silhouette_score(X, y_pred, metric="euclidean"))

        elbow_plot_distances.append(kmedoids.inertia_)
        
    c=2
    for i in silhouette_scores:
        print('Clusters = '+str(c)+'  Silhouette Score: %.3f' % i)
        c+=1
        

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, elbow_plot_distances, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Plot - K-medoids')
    plt.show()
    
    
    
    

# flatten 3D array to 2D pandas dataframe
def flatten_multivariate_time_series(data):
    # Get the shape of the input array
    num_stations, num_timestamps, num_features = data.shape

    # Create a list of column names based on your specified format
    column_names = [
        f"station{station}_feature{feature}"
        for station in range(num_stations)
        for feature in range(num_features)
    ]

    # Reshape the data into a 2D array
    reshaped_data = data.transpose(1, 0, 2).reshape(num_timestamps, -1)

    # Create a DataFrame with column names
    flattened_df = pd.DataFrame(reshaped_data, columns=column_names)

    return flattened_df


def predict_and_evaluate_all_stations_rmse(data_df):
    num_stations = len(data_df.columns) // 4
    total_rmse = 0.0
    if (num_stations>1):
        for station_index in range(num_stations):
            # Calculate the starting and ending column indexes for the current station
            start_col = station_index * 4
            end_col = (station_index + 1) * 4

            # Extract the data for the current station
            station_data = data_df.iloc[:, start_col:end_col]

            # Remove the data for the current station from the dataset
            reduced_df = data_df.drop(columns=station_data.columns)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                reduced_df, station_data, test_size=0.2, random_state=42
            )

            # Train a prediction model (e.g., Linear Regression)
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions on the test data
            predictions = model.predict(X_test)

            # Calculate the Root Mean Squared Error (RMSE) for the station
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            print(f"RMSE for station {station_index}: {rmse}")

            # Add the RMSE to the total
            total_rmse += rmse

        # Calculate the average RMSE across all stations
        average_rmse = total_rmse / num_stations
        #print(f"Average RMSE across all stations: {average_rmse}")
    else:
        average_rmse = total_rmse
    
    return average_rmse


def calculate_cluster_rmse(data_df, cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    total_cluster_rmse = 0.0
    
    for cluster_label in unique_clusters:
        # Find stations in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        print("Cluster "+str(cluster_label))
        # Extract the data for stations in the current cluster
        cluster_data = data_df.iloc[:, [i * 4 + j for i in cluster_indices for j in range(4)]]
        #print(cluster_data)
        # Calculate RMSE within the cluster
        cluster_rmse = predict_and_evaluate_all_stations_rmse(cluster_data)
        
        print(f"RMSE for Cluster {cluster_label}: {cluster_rmse}")
        
        # Add cluster RMSE to the total
        total_cluster_rmse += cluster_rmse

    # Calculate the average RMSE across all clusters
    average_cluster_rmse = total_cluster_rmse / len(unique_clusters)
    print(f"Average Cluster RMSE: {average_cluster_rmse}")
    return average_cluster_rmse
    

def plot_StAS_map_clustering_IDs(stations, cluster_labels, title):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    BBox = ((-87.649, -79.728, #longitude
          23.996, 31.062)) #latitude
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))

    
    

        
    for i in range(0,len(n_clusters)):
        if (len(n_clusters)==1):
            color = color_Cblack
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)
        else:
            if (i==0):
                color = color_C2
            elif (i==1):
                color = color_C1
            elif (i==2):
                color = color_C5
            elif (i==3):
                color = color_C3
            elif (i==4):
                color = color_C4
            elif (i==5):
                color = color_C6    

            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title(title)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    plt.rcParams.update({'font.size':9})
    
    for index, row in df_plot.iterrows():
        #st_name = row['StationName'].rsplit(' (')[0]
        st_id = row['StationID']
        if ((st_id == 290) | (st_id == 320) | (st_id == 302)):
            plt.text(row['longitude']+0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        elif (st_id == 360):
            plt.text(row['longitude']-0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        else: 
            plt.text(row['longitude'],row['latitude']-0.22, st_id, horizontalalignment='center', size='medium', color='black')
            
        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')

    
def plot_BAS_map_clustering_IDs(stations, cluster_labels, title):

    stations_coord = stations[['StationName','StationID','latitude', 'longitude']]
        
    df_plot = stations_coord.sort_values('StationID')

    BBox = ((-87.649, -79.728, #longitude
          23.996, 31.062)) #latitude
    
    df_plot['cluster'] = cluster_labels


    n_clusters=df_plot['cluster'].unique()

    dfs_clusters=[]

    for i in range(0,len(n_clusters)):
        df_temp = df_plot.loc[df_plot['cluster'] == n_clusters[i]]
        dfs_clusters.append(df_temp)

    florida_map = plt.imread('map.png')
    fig, ax = plt.subplots(figsize = (8,7))

    
    

        
    for i in range(0,len(n_clusters)):
        if (len(n_clusters)==1):
            color = color_Cblack
            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)
        else:
            if (i==0):
                color = color_C0
            elif (i==1):
                color = color_C1
            elif (i==2):
                color = color_C2
            elif (i==3):
                color = color_C3
            elif (i==4):
                color = color_C5
            elif (i==5):
                color = color_C4
            else:
                color = color_C6

            ax.scatter(dfs_clusters[i].longitude, dfs_clusters[i].latitude, zorder=1, alpha=1, c=color, s=40)

    
    ax.set_title(title)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])

    plt.rcParams.update({'font.size':9})
    
    for index, row in df_plot.iterrows():
        #st_name = row['StationName'].rsplit(' (')[0]
        st_id = row['StationID']
        if ((st_id == 290) | (st_id == 320) | (st_id == 302)):
            plt.text(row['longitude']+0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        elif ((st_id == 360) | (st_id == 380)):
            plt.text(row['longitude']-0.22,row['latitude']-0.06, st_id, horizontalalignment='center', size='medium', color='black')
        else: 
            plt.text(row['longitude'],row['latitude']-0.22, st_id, horizontalalignment='center', size='medium', color='black')
            
        
    ax.imshow(florida_map, zorder=0, extent = BBox, aspect= 'equal')
    

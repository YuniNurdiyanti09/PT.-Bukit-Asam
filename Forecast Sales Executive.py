
import numpy as np
import pandas as pd
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import itertools
import pyodbc
# from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error as smape
from dotenv.main import load_dotenv
import os
from datetime import datetime


# Nama Database yang digunakan
server = '10.3.4.139,1433'
database = 'dwh_prod'
username = 'developerptba'
password = 'LnfPYVFW1K2TAKf'

# untuk memanggil Data (harus terkoneksi dengan jaringan lokal)
cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
query = "SELECT * FROM dwh.DM_financial_position_unpivot;"
df = pd.read_sql(query, cnxn)

# preprocessing
# filter untuk mengambil data Total Sales
df_SALES = df[df['BreakDown'].isin(['Total Domestic Sales of Coal','Total Export Sales of Coal'])].copy()

# data yang diambil dalam prediksi adalah data aktual
df_SALES_actual = df_SALES[df_SALES['tipe'] == "Actual"]

# data diurutkan dari awal data
df_SALES_date = df_SALES_actual.sort_values(by='Date')

# groupby berdasarkan bulan yang terdia
df_SALES_group = df_SALES_date.groupby(['Date'])['amount'].sum().reset_index()

# dikarenakan peramalan time series menggunakan model forecasting prophet, maka kolom yang digunakan hanya tanggal dan tonase
# selain itu model hanya bisa bekerja jika kolom diubah menjadi ['ds','y']
df_SALES_group.columns = ['ds','y']

# tipe data diubah kedalam datetime
df_SALES_group['ds'] = pd.to_datetime(df_SALES_group['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.01, 0.05,0.1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.05, 1.0]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# fase tahap modeling dengan prophet + Tuning Parameter
# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_SALES_group))
train_data = df_SALES_group[:train_size]
test_data = df_SALES_group[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=False, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=False, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2]
)
final_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Melatih model dengan seluruh data
final_model.fit(df_SALES_group)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df = forecast_final[['ds', 'yhat']]

# Menyusun Data Frame  Total Sales
forecast_rows = pd.DataFrame({
    'Date': forecast_df['ds'],
    'Tonase': forecast_df['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Total Sales'
})

df_actual_sales = pd.DataFrame({
    'Date': df_SALES_group['ds'],
    'Tonase': df_SALES_group['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Total Sales'
})

# Menggabungkan semua DataFrame
result_df_Sales = pd.concat([df_actual_sales,forecast_rows ], ignore_index=True)

# ------------------------------------------------------------------------------------------
# preprocessing total sales export

# filter untuk data Total Sales Export
df_Sales_Ekspor = df_SALES_actual[df_SALES_actual['BreakDown']=='Total Export Sales of Coal']

# data diurutkan tanggal terdahulu
df_Sales_Ekspor = df_Sales_Ekspor.sort_values(by='Date')

# group data berdasarkan tanggal
df_Sales_Ekspor = df_Sales_Ekspor.groupby(['Date'])['amount'].sum().reset_index()

# memastikan data yang digunakan tidak bernilai 0 atau null
df_Sales_Ekspor = df_Sales_Ekspor[df_Sales_Ekspor['amount']!= 0]

# dikarenakan peramalan time series menggunakan model forecasting prophet, maka kolom yang digunakan hanya tanggal dan tonase
# selain itu model hanya bisa bekerja jika kolom diubah menjadi ['ds','y']
df_Sales_Ekspor.columns = ['ds','y']
df_Sales_Ekspor['ds'] = pd.to_datetime(df_Sales_Ekspor['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.01, 0.05,0.1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.05, 1.0]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Sales_Ekspor))
train_data = df_Sales_Ekspor[:train_size]
test_data = df_Sales_Ekspor[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        growth='linear'
    )
    model.add_seasonality(name='monthly', period=30, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2],
    growth='linear'
)
final_model.add_seasonality(name='monthly', period=30, fourier_order=3)

# Melatih model dengan seluruh data
final_model.fit(df_Sales_Ekspor)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_sales_ekspor = forecast_final[['ds', 'yhat']]

# membaut dataframe untuk Data total Sales Export
forecast_rows_SE = pd.DataFrame({
    'Date': forecast_df_sales_ekspor['ds'],
    'Tonase': forecast_df_sales_ekspor['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Total Sales Ekspor'
})

df_actual_sales_ekspor = pd.DataFrame({
    'Date': df_Sales_Ekspor['ds'],
    'Tonase': df_Sales_Ekspor['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Total Sales Ekspor'
})

# Menggabungkan semua DataFrame
result_df_Sales_Export = pd.concat([df_actual_sales_ekspor,forecast_rows_SE ], ignore_index=True)

# -----------------------------------------------------------------
# preprocessing untuk data total sales domestik
# untuk step selanjutnya hampir sama dengan preprocessing  dan model selection pada data Total sales
df_Sales_Domestic = df_SALES_actual[df_SALES_actual['BreakDown']=='Total Domestic Sales of Coal']
df_Sales_Domestic = df_Sales_Domestic.sort_values(by='Date')
df_Sales_Domestic = df_Sales_Domestic.groupby(['Date'])['amount'].sum().reset_index()
df_Sales_Domestic.columns = ['ds','y']
df_Sales_Domestic['ds'] = pd.to_datetime(df_Sales_Domestic['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [ 0.1, 0.5, 1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.1]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_Sales_Domestic))
train_data = df_Sales_Domestic[:train_size]
test_data = df_Sales_Domestic[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        growth='linear'
    )
    model.add_seasonality(name='monthly', period=30, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2],
    growth='linear'
)
final_model.add_seasonality(name='monthly', period=30, fourier_order=3)

# Melatih model dengan seluruh data
final_model.fit(df_Sales_Domestic)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_sales_df_Sales_Domestic = forecast_final[['ds', 'yhat']]

# Menambahkan data untuk 'Forecast'
forecast_rows_SE = pd.DataFrame({
    'Date': forecast_df_sales_df_Sales_Domestic['ds'],
    'Tonase': forecast_df_sales_df_Sales_Domestic['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Total Sales Domestik'
})

df_actual_sales_forecast_df_sales_df_Sales_Domestic = pd.DataFrame({
    'Date': df_Sales_Domestic['ds'],
    'Tonase': df_Sales_Domestic['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Total Sales Domestik'
})

# Menggabungkan semua DataFrame
result_df_SD = pd.concat([df_actual_sales_forecast_df_sales_df_Sales_Domestic,forecast_rows_SE], ignore_index=True)
result_df_SD

# -------------------------------------------------------------------
# Preprocessing untuk data Average selling price
# step by step hampir sama dengan code diatas
df_ASP = df[df['BreakDown']=='Average Market Price IDR']
df_ASP =df_ASP[df_ASP['tipe']=='Actual']
df_ASP = df_ASP.sort_values(by='Date')
df_ASP = df_ASP.groupby(['Date'])['amount'].sum().reset_index()
df_ASP.columns = ['ds','y']
df_ASP['ds'] = pd.to_datetime(df_ASP['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [ 0.1, 0.5, 1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.1]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_ASP))
train_data = df_ASP[:train_size]
test_data = df_ASP[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        growth='linear'
    )
    model.add_seasonality(name='monthly', period=30, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2],
    growth='linear'
)
final_model.add_seasonality(name='monthly', period=30, fourier_order=3)

# Melatih model dengan seluruh data
final_model.fit(df_ASP)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_df_ASP = forecast_final[['ds', 'yhat']]

# Menambahkan data untuk 'Forecast'
forecast_rows_SE = pd.DataFrame({
    'Date': forecast_df_ASP['ds'],
    'Tonase': forecast_df_ASP['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Avg. Selling Price'
})

df_actual_sales_forecast_df_ASP = pd.DataFrame({
    'Date': df_ASP['ds'],
    'Tonase': df_ASP['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Avg. Selling Price'
})

# Menggabungkan semua DataFrame
result_df_Avg_SP = pd.concat([df_actual_sales_forecast_df_ASP,forecast_rows_SE ], ignore_index=True)
result_df_Avg_SP

# --------------------------------------------------------------------

df_ASP_Domestic = df[df['BreakDown']=='Average Domestic Price']
df_ASP_Domestic = df_ASP_Domestic[df_ASP_Domestic['tipe']=='Actual']
df_ASP_Domestic = df_ASP_Domestic.sort_values(by='Date')
df_ASP_Domestic = df_ASP_Domestic.groupby(['Date'])['amount'].sum().reset_index()

df_ASP_Domestic.columns = ['ds','y']
df_ASP_Domestic['ds'] = pd.to_datetime(df_ASP_Domestic['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [0.01, 0.02]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.05, 0.1]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_ASP_Domestic))
train_data = df_ASP_Domestic[:train_size]
test_data = df_ASP_Domestic[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        growth='linear'
    )
    model.add_seasonality(name='monthly', period=30, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2],
    growth='linear'
)
final_model.add_seasonality(name='monthly', period=30, fourier_order=3)

# Melatih model dengan seluruh data
final_model.fit(df_ASP_Domestic)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_ASP_Domestic = forecast_final[['ds', 'yhat']]

# Menambahkan data untuk 'Forecast'
forecast_rows_SE = pd.DataFrame({
    'Date': forecast_ASP_Domestic['ds'],
    'Tonase': forecast_ASP_Domestic['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Avg. Selling Price Domestic'
})

df_actual_sales_forecast_df_ASP_Domestic = pd.DataFrame({
    'Date': df_ASP_Domestic['ds'],
    'Tonase': df_ASP_Domestic['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Avg. Selling Price Domestic'
})

# Menggabungkan semua DataFrame
result_df_Avg_SP_Dom = pd.concat([df_actual_sales_forecast_df_ASP_Domestic,forecast_rows_SE], ignore_index=True)
result_df_Avg_SP_Dom

# ------------------------------------------------------------------------
# preprocessing untuk data Average Selling Price Export

df_ASP_Export = df[df['BreakDown']=='Average Export Price']
df_ASP_Export = df_ASP_Export[df_ASP_Export['tipe']=='Actual']
df_ASP_Export = df_ASP_Export.sort_values(by='Date')
df_ASP_Export = df_ASP_Export.groupby(['Date'])['amount'].sum().reset_index()

df_ASP_Export.columns = ['ds','y']
df_ASP_Export['ds'] = pd.to_datetime(df_ASP_Export['ds'])

# Kombinasi parameter untuk tuning
changepoint_prior_scales = [ 0.1, 0.5, 1]
seasonality_modes = ['additive']
seasonality_prior_scales = [0.01, 0.1]

# Membuat list kombinasi parameter
param_combinations = list(itertools.product(changepoint_prior_scales, seasonality_modes, seasonality_prior_scales))

# Fungsi untuk menghitung SMAPE
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))

# Menyimpan hasil tuning
best_smape = float('inf')
best_params = None

# Train dan test split
train_size = int(0.8 * len(df_ASP_Export))
train_data = df_ASP_Export[:train_size]
test_data = df_ASP_Export[train_size:]

# Looping untuk setiap kombinasi parameter
for changepoint_prior_scale, seasonality_mode, seasonality_prior_scale in param_combinations:
    
    # Inisialisasi model Prophet dengan parameter tuning
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale,
        growth='linear'
    )
    model.add_seasonality(name='monthly', period=30, fourier_order=3)
    
    # Melatih model dengan data train
    model.fit(train_data)

    # Membuat dataframe masa depan yang mencakup periode data test
    future = model.make_future_dataframe(periods=len(test_data), freq='MS')

    # Melakukan prediksi
    forecast = model.predict(future)

    # Mengambil prediksi yang sesuai dengan data test
    forecast_test = forecast[-len(test_data):]

    # Menggabungkan data asli dengan prediksi
    test_data.loc[:, 'yhat'] = forecast_test['yhat'].values

    # Menghitung SMAPE untuk data test
    smape_value = smape(test_data['y'], test_data['yhat'])
    print(f"SMAPE dengan changepoint_prior_scale={changepoint_prior_scale}, seasonality_mode={seasonality_mode}, seasonality_prior_scale={seasonality_prior_scale}: {smape_value:.2f}%")

    # Menyimpan kombinasi parameter terbaik
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = (changepoint_prior_scale, seasonality_mode, seasonality_prior_scale)

print(f"\nParameter terbaik: changepoint_prior_scale={best_params[0]}, seasonality_mode={best_params[1]}, seasonality_prior_scale={best_params[2]}")
print(f"SMAPE terbaik: {best_smape:.2f}%")

# Latih model dengan seluruh data menggunakan parameter terbaik
final_model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True, 
    daily_seasonality=False,
    changepoint_prior_scale=best_params[0],
    seasonality_mode=best_params[1],
    seasonality_prior_scale=best_params[2],
    growth='linear'
)
final_model.add_seasonality(name='monthly', period=30, fourier_order=3)

# Melatih model dengan seluruh data
final_model.fit(df_ASP_Export)

# Membuat dataframe masa depan untuk 6 bulan ke depan
future = final_model.make_future_dataframe(periods=12, freq='MS')

# Melakukan prediksi
forecast_final = final_model.predict(future)

# Menyimpan hasil prediksi ke dalam DataFrame
forecast_ASP_Export = forecast_final[['ds', 'yhat']]

# Menambahkan data untuk 'Forecast'
forecast_rows_SE = pd.DataFrame({
    'Date':forecast_ASP_Export['ds'],
    'Tonase': forecast_ASP_Export['yhat'],
    'Jenis': 'Forecast',
    'Breakdown' : 'Avg. Selling Price Export'
})

df_actual_sales_forecast_df_ASP_Export = pd.DataFrame({
    'Date': df_ASP_Export['ds'],
    'Tonase': df_ASP_Export['y'],
    'Jenis': 'Actual',
    'Breakdown' : 'Avg. Selling Price Export'
})


# Menggabungkan semua DataFrame
result_df_Avg_SP_Exp = pd.concat([df_actual_sales_forecast_df_ASP_Export,forecast_rows_SE ], ignore_index=True)
result_df_Avg_SP_Exp

# -----------------------------------------------------
# kumpulan data Actual dan Forecast yang sudah menjadi dataframe

df_Final = pd.concat((result_df_Sales, result_df_SD, result_df_Sales_Export, result_df_Avg_SP, result_df_Avg_SP_Dom, result_df_Avg_SP_Exp), axis = 0)
# melakukan reset index agar tidak berantakan

df_Final = df_Final.reset_index(drop=True)
# preprocessing untuk mengam
# DF Plan
df_plan = df[df['tipe']=='RKAP'].copy()

df_plan_sales = df_plan[df_plan['BreakDown'].isin(['Total Domestic Sales of Coal','Total Export Sales of Coal'])].copy()
df_plan_sales = df_plan_sales.groupby(['Date'])['amount'].sum().reset_index()
df_plan_sales_domestik = df_plan[df_plan['BreakDown'] == 'Total Domestic Sales of Coal'].groupby(['Date'])['amount'].sum().reset_index()
df_plan_sales_export = df_plan[df_plan['BreakDown'] == 'Total Export Sales of Coal'].groupby(['Date'])['amount'].sum().reset_index()
df_plan_ASP_Export = df_plan[df_plan['BreakDown'] == 'Average Export Price'].groupby(['Date'])['amount'].sum().reset_index()
df_plan_ASP = df_plan[df_plan['BreakDown'] == 'Average Market Price IDR'].groupby(['Date'])['amount'].sum().reset_index()
df_plan_ASP_Domestic = df_plan[df_plan['BreakDown'] == 'Average Domestic Price'].groupby(['Date'])['amount'].sum().reset_index()

# Menambahkan data untuk 'Forecast'
df_plan_sales = pd.DataFrame({
    'Date':df_plan_sales['Date'],
    'Tonase': df_plan_sales['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Total Sales'
})

df_plan_sales_domestik = pd.DataFrame({
    'Date': df_plan_sales_domestik['Date'],
    'Tonase': df_plan_sales_domestik['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Total Sales Domestik'
})
df_plan_sales_export = pd.DataFrame({
    'Date': df_plan_sales_export['Date'],
    'Tonase': df_plan_sales_export['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Total Sales Ekspor'
})
df_plan_ASP = pd.DataFrame({
    'Date': df_plan_ASP['Date'],
    'Tonase': df_plan_ASP['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Avg. Selling Price'
})
df_plan_ASP_Export = pd.DataFrame({
    'Date': df_plan_ASP_Export['Date'],
    'Tonase': df_plan_ASP_Export['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Avg. Selling Price Export'
})
df_plan_ASP_Domestic = pd.DataFrame({
    'Date': df_plan_ASP_Domestic['Date'],
    'Tonase': df_plan_ASP_Domestic['amount'],
    'Jenis': 'Plan',
    'Breakdown' : 'Avg. Selling Price Domestic'
})
new_row = pd.DataFrame({
    'Date': [datetime.now()],  # Menggunakan datetime sekarang
    'Tonase': [0],  # Tonase diisi 0
    'Jenis': ['Actual'],  # Jenis diisi "Actual"
    'Breakdown': ['Last Update']  # Breakdown diisi "Last Update"
})

# Menggabungkan semua DataFrame
df_All = pd.concat([df_plan_sales, df_plan_sales_domestik, df_plan_sales_export, df_plan_ASP, df_plan_ASP_Domestic, df_plan_ASP_Export, df_Final, new_row ], ignore_index=True)

df_All['Date'] = pd.to_datetime(df_All['Date'])
df_All['Tonase'] = round(df_All['Tonase'], 2)

# -------------------------------------------------
# deployement hasil model forecast kedalam Data mart (DM_forecasting_sales)
# Koneksi ke SQL Server

server = '10.3.4.139,1433'
database = 'dwh_prod'
username = 'developerptba'
password = 'LnfPYVFW1K2TAKf'

# Inisialisasi koneksi ke SQL Server
try:
    conn = pyodbc.connect(
        f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )
    print("Koneksi ke database berhasil.")
except Exception as e:
    print(f"Error saat mencoba koneksi ke database: {e}")
    exit()

cursor = conn.cursor()

# Proses truncate tabel
truncate_query = "TRUNCATE TABLE dwh.DM_Sales_Forecasting"
try:
    cursor.execute(truncate_query)
    conn.commit()
    print("Tabel berhasil di-truncate.")
except Exception as e:
    print(f"Error saat truncate tabel: {e}")
    conn.close()
    exit()


# Pastikan `Df_Final` sudah didefinisikan
if 'df_All' not in globals():
    print("Error: DataFrame `df_All` tidak ditemukan.")
    conn.close()
    exit()

# Menyiapkan query untuk insert
insert_query = """
    INSERT INTO dwh.DM_Sales_Forecasting 
    ([Date], [Tonase], [Jenis], [Breakdown]) 
    VALUES (?, ?, ?, ?)
"""

rows_to_insert = [tuple(row) for row in df_All.itertuples(index=False)]

# Proses insert data
cursor.fast_executemany = True
try:
    cursor.executemany(insert_query, rows_to_insert)
    conn.commit()
    print(f"Berhasil Insert {len(rows_to_insert)} baris data.")
except Exception as e:
    conn.rollback()
    print(f"Error saat menginsert data: {e}")

# Menutup koneksi
cursor.close()
conn.close()
print("Forecasting Done")

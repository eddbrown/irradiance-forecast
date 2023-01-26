import pandas as pd
import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from multiprocessing import cpu_count, Pool

class IrradianceDataset(Dataset):
    def __init__(self, dates, image_folder, irradiance_file, channel='0211', scaler=None, forecast_horizon_hours=0):

        self.image_folder = image_folder
        self.irradiance_data = pd.read_hdf(irradiance_file)
        self.irradiance_data = self.irradiance_data.loc[(self.irradiance_data>=1).all(axis=1)]
        self.dates = dates
        self.channel = channel
        self.forecast_horizon_hours = forecast_horizon_hours
        print('Checking available data...')
        with Pool(cpu_count()) as pool:
            check_dates = list(pool.map(self.check_date, self.dates))
        self.dates = sorted([date for check, date in check_dates if check])
#         self.dates = [date for date in tqdm(self.dates) if self.check_date(date)]
        self.irradiance_data = self.irradiance_data.loc[self.dates,:]
        if scaler is None:
            self.scaler = QuantileTransformer(n_quantiles=1000)
            self.scaler.fit(self.irradiance_data)
        else:
            self.scaler = scaler
        self.scaled_irradiance_data = self.scaler.transform(self.irradiance_data)
        print('Dataset Length:', len(self.dates))
        
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, i):
        forecast_date = self.dates[i]
        image_date = forecast_date - pd.Timedelta(hours=self.forecast_horizon_hours)
        image_file_name = self.get_file_name(image_date)
        irradiance_data = torch.FloatTensor(self.scaled_irradiance_data[i])
       
        return self.load_image(image_file_name), irradiance_data
    
    def check_date(self, date):
        image_date = date - pd.Timedelta(hours=self.forecast_horizon_hours)
        image_file_name = self.get_file_name(image_date)
        
        if not os.path.exists(image_file_name):
            return False, date
        try:
            irradiance_datum = self.irradiance_data.loc[date]
        except:
            return False, date
        return True, date
        
    def load_image(self, file_path):
        with open(file_path, 'rb') as f:
            image = np.load(f)['x']
        image = self.scale(image)
        return torch.Tensor(image).unsqueeze(0)
    
    def scale(self, image):
        if self.channel == '0211':
            image = np.log(np.clip(image, 5, 2000))
            image = (image - np.log(5))/(np.log(2000) - np.log(5))
        return image
        
    def get_file_name(self, date):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        hour = str(date.hour).zfill(2)
        minute = str(date.minute).zfill(2)
        file_path = f'{self.image_folder}/{year}/{month}/{day}/AIA{year}{month}{day}_{hour}{minute}_{self.channel}.npz'
        return file_path
import pandas as pd
import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from multiprocessing import cpu_count, Pool
import tqdm

class IrradianceDataset(Dataset):
    def __init__(self, dates, image_folder, irradiance_file, channels=['0211'], scaler=None, forecast_horizon_hours=0, flip_augment=True):
        self.image_folder = image_folder
        self.irradiance_data = pd.read_hdf(irradiance_file)
        self.irradiance_data = self.irradiance_data.loc['2010-05-01 00:00:00':'2018-12-31 23:00:00', :]
        self.flip_augment = flip_augment
        self.dates = dates
        self.channels = channels
        self.forecast_horizon_hours = forecast_horizon_hours
        
        # Remove outliers
        print('Irradiance Dataset: Dataset length before removing zeros:', len(self.irradiance_data))
        self.irradiance_data = self.irradiance_data.loc[(self.irradiance_data>=1).all(axis=1)]
        print('Irradiance Dataset: Dataset length after removing zeros:', len(self.irradiance_data))
        
        print('Irradiance Dataset: Dataset length before removing outliers:', len(self.irradiance_data))
        medians = self.irradiance_data.median()
        for i, column in enumerate(self.irradiance_data.columns):
            self.irradiance_data = self.irradiance_data[self.irradiance_data[column] < 1_000 * medians[i]]
        print('Irradiance Dataset: Dataset length after removing outliers:', len(self.irradiance_data))

        print('Irradiance Dataset: Checking available data...')
        check_dates = [self.check_date(date) for date in self.dates]
        self.dates = sorted([date for check, date in check_dates if check])
        self.irradiance_data = self.irradiance_data.loc[self.dates,:]
        
        print('Irradiance Dataset: Scaling data...')
        if scaler is None:
            self.scaler = QuantileTransformer(n_quantiles=1000)
            self.scaler.fit(self.irradiance_data)
        else:
            self.scaler = scaler
        self.scaled_irradiance_data = self.scaler.transform(self.irradiance_data)
        
        print('Irradiance Dataset: Final Dataset Length:', len(self.dates))
        
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, i):
        forecast_date = self.dates[i]
        image_date = forecast_date - pd.Timedelta(hours=self.forecast_horizon_hours)
        images = []
        for channel in self.channels:
            image_file_name = self.get_file_name(image_date, channel)
            images.append(self.load_image(image_file_name, channel))
            
        images = torch.cat(images, dim=1)
        
        scaled_irradiance_data = torch.FloatTensor(self.scaled_irradiance_data[i])
       
        return images, scaled_irradiance_data
    
    def check_date(self, date):
        image_date = date - pd.Timedelta(hours=self.forecast_horizon_hours)
        
        for channel in self.channels:
            image_file_name = self.get_file_name(image_date, channel)
        
            if not os.path.exists(image_file_name):
                return False, date
            try:
                irradiance_datum = self.irradiance_data.loc[date]
            except:
                return False, date
        return True, date
        
    def load_image(self, file_path, channel):
        with open(file_path, 'rb') as f:
            image = np.load(f)['x']
        if self.flip_augment:
            if np.random.choice([True, False]):
                image = np.flipud(image)
        image = self.scale(image, channel)
        return torch.Tensor(image).unsqueeze(0)
    
    def scale(self, image, channel):
        if channel == '0211':
            image = np.log(np.clip(image, 5, 20000))
            image = (image - np.log(5))/(np.log(20000) - np.log(5))
        elif channel == '0094':
            image = np.log(np.clip(image, 5, 1000))
            image = (image - np.log(5))/(np.log(1000) - np.log(5))
        elif channel == '0335':
            image = np.log(np.clip(image, 5, 2000))
            image = (image - np.log(5))/(np.log(2000) - np.log(5))
        elif channel == '1600':
            image = np.log(np.clip(image, 5, 3000))
            image = (image - np.log(5))/(np.log(3000) - np.log(5))
        return image
        
    def get_file_name(self, date, channel):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        hour = str(date.hour).zfill(2)
        minute = str(date.minute).zfill(2)
        file_path = f'{self.image_folder}/{year}/{month}/{day}/AIA{year}{month}{day}_{hour}{minute}_{channel}.npz'
        return file_path
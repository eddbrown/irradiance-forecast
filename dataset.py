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
        self.irradiance_data = self.irradiance_data.loc['2010-05-01 00:00:00':'2018-12-31 23:30:00', :]
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
        
        print('Irradiance Dataset: Data available with selected dates:', len(self.dates))
        print('Irradiance Dataset: Scaling data...')
        if scaler is None:
            self.scaler = QuantileTransformer(n_quantiles=1000)
            self.scaler.fit(self.irradiance_data)
        else:
            self.scaler = scaler
        self.scaled_irradiance_data = self.irradiance_data.copy()
        self.scaled_irradiance_data[self.irradiance_data.columns] = self.scaler.transform(self.irradiance_data)
        
        print('Irradiance Dataset: Final Dataset Length:', len(self.dates))
        
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, i):
        forecast_date = self.dates[i]
        image_date = forecast_date - pd.Timedelta(hours=self.forecast_horizon_hours)
        image_date = pd.to_datetime(str(image_date))
        images = []
        for channel in self.channels:
            image_file_name = self.get_file_name(image_date, channel)
            images.append(self.load_image(image_file_name, channel))
            
        images = torch.cat(images, dim=1)
        
        irradiance_data_forecast_date = torch.FloatTensor(
            self.scaled_irradiance_data.loc[forecast_date,:])
        irradiance_data_image_date = torch.FloatTensor(
            self.scaled_irradiance_data.loc[image_date,:])
       
        return images, irradiance_data_image_date, irradiance_data_forecast_date
    
    def check_date(self, date):
        image_date = pd.to_datetime(date - pd.Timedelta(hours=self.forecast_horizon_hours))
        
        for channel in self.channels:
            image_file_name = self.get_file_name(image_date, channel)
        
            if not os.path.exists(image_file_name):
                return False, date
            try:
                irradiance_datum = self.irradiance_data.loc[date,:]
                irradiance_datum = self.irradiance_data.loc[image_date,:]
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
        # Numbers picked by using the 20th and 99.999th percentile pixel values
        # of 100 random images from each channel
        # 20th percentile because we want to make most of the off-disk pixels the same as they are likely not relevant.
        if channel == '0094':
            return self.log_linear_scale(image, 0.52, 781.95)
        elif channel == '0211':
            return self.log_linear_scale(image, 18.02, 9264.69)
        elif channel == '0335':
            return self.log_linear_scale(image, 2.1, 946.49)
        elif channel == '1600':
            return self.log_linear_scale(image, 5.04, 1367.08)
        return image
    
    def log_linear_scale(self, image, min_, max_):
        image = np.log(np.clip(image, min_, max_))
        image = (image - np.log(min_))/(np.log(max_) - np.log(min_))
        return image
        
    def get_file_name(self, date, channel):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        hour = str(date.hour).zfill(2)
        minute = str(date.minute).zfill(2)
        file_path = f'{self.image_folder}/{year}/{month}/{day}/AIA{year}{month}{day}_{hour}{minute}_{channel}.npz'
        return file_path
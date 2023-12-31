import pandas as pd
import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from multiprocessing import cpu_count, Pool
import tqdm

POSSIBLE_CHANNELS = [
    '0094',
    '0131',
    '0171',
    '0193',
    '0211',
    '0304',
    '0335',
    '1600',
    '1700',
]

class CustomScaler():
    def __init__(self):
        self.min_max_scaler = MinMaxScaler()
        
    def fit(self, data):
        logged_data = np.log(data)
        self.min_max_scaler.fit(logged_data)

    def transform(self, data):
        logged_data = np.log(data)
        scaled_logged_data = self.min_max_scaler.transform(logged_data)
        return scaled_logged_data
    
    def inverse_transform(self, data):
        unscaled_data = self.min_max_scaler.inverse_transform(data)
        unlogged_data = np.exp(unscaled_data)
        return unlogged_data

class IrradianceDataset(Dataset):
    def __init__(self, dates, image_folder, irradiance_file, channels=['0211'], scaler=None, forecast_horizon_hours=0, flip_augment=False, min_date='2010-05-01 00:00:00', max_date='2018-12-31 23:30:00'):
        self.image_folder = image_folder
        self.irradiance_data = pd.read_hdf(irradiance_file)
        
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
        print('Irradiance Dataset: Scaling data...')
        if scaler is None:
            self.scaler = CustomScaler()
            self.scaler.fit(self.irradiance_data)
        else:
            self.scaler = scaler
        self.irradiance_data = self.irradiance_data.loc[min_date:max_date, :]

        print('Irradiance Dataset: Checking available data...')
        with Pool(cpu_count()) as pool:
            check_dates = pool.map(self.check_date, self.dates)
            
        self.dates = sorted([date for check, date in check_dates if check])
        
        print('Irradiance Dataset: Data available with selected dates:', len(self.dates))


        self.scaled_irradiance_data = self.irradiance_data.copy()
        self.scaled_irradiance_data[self.irradiance_data.columns] = self.scaler.transform(self.irradiance_data[self.irradiance_data.columns].values)
        
        print('Irradiance Dataset: Final Dataset Length:', len(self.dates))
        
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, i):
        forecast_date = self.dates[i]
        image_date = forecast_date - pd.Timedelta(hours=self.forecast_horizon_hours)
        image_date = pd.to_datetime(str(image_date))
        images = []
        
        # Have to flip before loading images to make sure all channels are flipped
        # the same way.
        if self.flip_augment:
            flip = np.random.choice([True, False])
        else:
            flip=False
                
        for channel in self.channels:
            image_file_name = self.get_file_name(image_date, channel)
            images.append(self.load_image(image_file_name, channel, flip=flip))
            
        images = torch.cat(images, dim=0)
        
        irradiance_data_forecast_date = torch.FloatTensor(
            self.scaled_irradiance_data.loc[forecast_date,:])
        irradiance_data_image_date = torch.FloatTensor(
            self.scaled_irradiance_data.loc[image_date,:])
       
        return images, irradiance_data_image_date, irradiance_data_forecast_date
    
    def check_date(self, date):
        if date not in self.irradiance_data.index:
            return False, date
        
        image_date = pd.to_datetime(date - pd.Timedelta(hours=self.forecast_horizon_hours))
        if image_date not in self.irradiance_data.index:
            return False, date
        
        for channel in POSSIBLE_CHANNELS:
            image_file_name = self.get_file_name(image_date, channel)
            if not os.path.exists(image_file_name):
                return False, date

        return True, date
        
    def load_image(self, file_path, channel,flip=True):
        with open(file_path, 'rb') as f:
            image = np.load(f)['x']
        if flip:
            image = np.flipud(image)
        image = self.scale(image, channel)
        return torch.Tensor(image).unsqueeze(0)
    
    def scale(self, image, channel):
        # Numbers picked by using the 20th and 99.999th percentile pixel values
        # of 100 random images from each channel
        # 20th percentile because we want to make most of the off-disk pixels the same as they are likely not relevant.
        if channel == '0094':
            return self.log_linear_scale(image, 0.52, 781.95)
        elif channel == '0131':
            return self.log_linear_scale(image, 1.25, 1482.33)
        elif channel == '0171':
            return self.log_linear_scale(image, 30.7, 13941.28)
        elif channel == '0193':
            return self.log_linear_scale(image, 57.57, 20292.16)
        elif channel == '0211':
            return self.log_linear_scale(image, 18.02, 9264.69)
        elif channel == '0304':
            return self.log_linear_scale(image, 6.43, 7293.93)
        elif channel == '0335':
            return self.log_linear_scale(image, 2.1, 946.49)
        elif channel == '1600':
            return self.log_linear_scale(image, 5.04, 1367.08)
        elif channel == '1700':
            return self.log_linear_scale(image, 60.4 , 12004.73)
        elif channel == 'bx':
            return self.linear_scale(image, -3000, 3000)
        elif channel == 'by':
            return self.linear_scale(image, -3000, 3000)
        elif channel == 'bz':
            return self.linear_scale(image, -3000, 3000)
        return image
    
    def log_linear_scale(self, image, min_, max_):
        image = np.log(np.clip(image, min_, max_))
        image = (image - np.log(min_))/(np.log(max_) - np.log(min_))
        return image
    
    def linear_scale(self, image, min_, max_):
        image = (image - min_)/(max_ - min_)
        
    def get_file_name(self, date, channel):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        hour = str(date.hour).zfill(2)
        minute = str(date.minute).zfill(2)
        if channel[0] == 'b':
            file_path = f'{self.image_folder}/{year}/{month}/{day}/HMI{year}{month}{day}_{hour}{minute}_{channel}.npz'
        else:
            file_path = f'{self.image_folder}/{year}/{month}/{day}/AIA{year}{month}{day}_{hour}{minute}_{channel}.npz'
        return file_path

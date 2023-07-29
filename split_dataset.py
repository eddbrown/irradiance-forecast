import numpy as np
from tqdm import tqdm
import pandas as pd

def split_dataset(min_date='2010-05-01', max_date='2019-01-01 00:00:00', buffer_days=2, freq='30T'):
    train_period = 30 * 24 * 2 #Freq is 30 mins so this is 30 days  
    validation_period = 5 * 24 * 2
    test_period = 5 * 24 * 2
    buffer_period = buffer_days * 24 * 2
    
    categories = ['train'] * train_period \
        + ['buffer'] * buffer_period \
        + ['validation'] * validation_period \
        + ['buffer'] * buffer_period \
        + ['test'] * test_period \
        + ['buffer'] * buffer_period
    
    dates = list(pd.date_range(start=min_date, end=max_date, freq=freq))
    
    train_dates = []
    validation_dates = []
    test_dates = []
    
    for i, date in enumerate(dates):
        category = categories[i % len(categories)]
        if category == 'train':
            train_dates.append(date)
        if category == 'validation':
            validation_dates.append(date)
        if category == 'test':
            test_dates.append(date)
            
    return train_dates, validation_dates, test_dates

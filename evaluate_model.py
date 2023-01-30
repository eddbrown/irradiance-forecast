from sklearn.metrics import mean_squared_error as mse
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from torch import nn

def evaluate_model(model, dataset, batch_size=32, device='cuda:0', num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    predictions = []
    targets = []
    losses = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            image, target = data[0], data[1]
            model_output = model.forward(image.to(device))
            predictions.append(model_output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    loss = mse(predictions, targets)
    
    predictions = dataset.scaler.inverse_transform(predictions)
    results = {}
    for band_index, band in enumerate(dataset.irradiance_data.columns):
        results['loss'] = loss
        persistence_prediction = dataset.irradiance_data[band].shift(freq=f'{60*dataset.forecast_horizon_hours}T')
        persistence_mape =  100*np.abs((persistence_prediction - dataset.irradiance_data[band]).div(dataset.irradiance_data[band])).dropna().mean()
       
        model_mape = mape(predictions[:, band_index], dataset.irradiance_data[band])
        
        if persistence_mape > 0:
            persistence_score = 1.0 - (model_mape/persistence_mape)
        else:
            persistence_score = 0.0
        results[f'rmse_{band}'] = rmse(predictions[:, band_index], dataset.irradiance_data[band])
        results[f'mape_{band}'] = mape(predictions[:, band_index], dataset.irradiance_data[band])
        results[f'persistence_mape_{band}'] = persistence_mape
        results[f'persistence_score_{band}'] = persistence_score
    
    return results 
    
    
def rmse(predictions, targets):
    return mse(predictions, targets, squared=False)

def mape(predictions, targets):
    return 100*np.mean(np.abs(np.divide((predictions - targets), targets)))
    
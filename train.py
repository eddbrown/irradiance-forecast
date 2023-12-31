import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from models import SolarIrradianceForecast, SolarIrradianceForecastAllChannels
from dataset import IrradianceDataset
import wandb
import argparse
import time
import json
from tqdm import tqdm
from split_dataset import split_dataset
from evaluate_model import evaluate_model
import numpy as np
import sys
torch.multiprocessing.set_sharing_strategy('file_system')

def train():    
    parser = argparse.ArgumentParser(description='Solar Irradiance Forecasts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--learning_rate',default=1e-4, type=float)
    parser.add_argument('--image_folder', default='/data/hpcdata/users/edbrown41/SDOMLnpz256', type=str)
    parser.add_argument('--irradiance_data_file', default="/data/hpcdata/users/edbrown41/stan_bands.h5", type=str)
    parser.add_argument('--run_name', default='run', type=str)
    parser.add_argument('--random_seed', default=999, type=int)
    parser.add_argument('--save_folder', default='/data/hpcdata/users/edbrown41/irradiance-forecast/runs', type=str)
    parser.add_argument('--key_file', default='/data/hpcdata/users/edbrown41/irradiance-forecast/keys.json', type=str)
    parser.add_argument('--run_description', help='More verbose run description to describe the run', type=str, required=True)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--forecast_horizon_hours', default=0, type=int)
    parser.add_argument('--flip_augment', action='store_true')
    parser.add_argument('--channels', default='0211', type=str)
    parser.add_argument('--loss_function', default='mse', type=str)
    parser.add_argument('--add_persistence', action='store_true')
    parser.add_argument('--min_date', default='2010-05-01 00:00:00', type=str)
    parser.add_argument('--max_date', default='2018-12-31 23:30:00', type=str)
    parser.add_argument('--model_type', default='vision_transformer', type=str)
    
    config = parser.parse_args()
    config.channels = config.channels.split(',')
    
    # Set random seed for reproducibility
    print("Random Seed: ", config.random_seed)
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    time_now = pd.to_datetime(pd.Timestamp.now()).strftime('%Y-%m-%d %H:%M:%S')
    specific_run_name = f'{config.run_name}_{time_now}'
    config.specific_run_name = specific_run_name
    config.date = str(time_now)
    
    with open(config.key_file) as json_file:
        wandb_api_key = json.load(json_file)['api_key']
    wandb.login(key=wandb_api_key, force=True)
    wandb.init(project='new-irradiance_forecast', config=vars(config), entity="eddyb92")
    
    print(config)
    wandb.run.name = specific_run_name
    
    # Half the memory of float64
    torch.set_default_dtype(torch.float32)
    
    run_save_folder = os.path.join(config.save_folder, specific_run_name)
    checkpoint_folder = os.path.join(run_save_folder, 'run_checkpoints')
    if not os.path.exists(run_save_folder):
        os.mkdir(run_save_folder)
        os.mkdir(checkpoint_folder)
        
    with open(os.path.join(run_save_folder, 'config.json'), 'w') as f:
        json.dump(vars(config), f)
        
    device = 'cuda:0' if config.gpus >= 1 else 'cpu'

    losses = []
    
    train_dates, validation_dates, test_dates = split_dataset(min_date=config.min_date, max_date=config.max_date)

    train_dataset = IrradianceDataset(
        list(train_dates),
        config.image_folder,
        config.irradiance_data_file,
        channels=config.channels,
        forecast_horizon_hours=config.forecast_horizon_hours,
        flip_augment=config.flip_augment
    )
    
    validation_dataset = IrradianceDataset(
        list(validation_dates),
        config.image_folder,
        config.irradiance_data_file,
        channels=config.channels,
        forecast_horizon_hours=config.forecast_horizon_hours,
        scaler=train_dataset.scaler
    )
    
    test_dataset = IrradianceDataset(
        list(test_dates),
        config.image_folder,
        config.irradiance_data_file,
        channels=config.channels,
        forecast_horizon_hours=config.forecast_horizon_hours,
        scaler=train_dataset.scaler
    )
    
    if config.model_type == 'vision_transformer':
        model = SolarIrradianceForecast(
            persistence=config.add_persistence
        ).to(device)
        
        if config.checkpoint != '':
            model_data = torch.load(config.checkpoint)
            model.load_state_dict(model_data['model'])
    elif config.model_type == 'all':
        model = SolarIrradianceForecastAllChannels(
            persistence=config.add_persistence
        ).to(device)
        
        if config.checkpoint != '':
            model_data = torch.load(config.checkpoint)
            model.load_state_dict(model_data['model'])


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training Loop
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True, shuffle=True)
    print("Starting Training Loop...")
    # For each epoch
    best_validation_loss = np.inf
    for epoch in range(config.num_epochs):
        for i, data in enumerate(tqdm(dataloader)):
            model.zero_grad()
            image, persistence, target = data[0], data[1], data[2]
            prediction = model.forward(image.to(device), persistence.to(device))
            loss = criterion(prediction, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            wandb.log({'loss': loss.item()})
            

        validation_results = evaluate_model(model,
                                            validation_dataset,
                                            config.batch_size,
                                            device,
                                            config.num_workers)

        test_results = evaluate_model(model,
                                      test_dataset,
                                      config.batch_size,
                                      device,
                                      config.num_workers)
        
        wandb.log({'validation_loss': validation_results['loss']})
        wandb.log({'validation_results': validation_results})
        wandb.log({'test_results': test_results})
                  
        torch.save(
            {
                'model': model.state_dict(),
                'scaler': train_dataset.scaler
            },
            os.path.join(checkpoint_folder, f'checkpoint_{epoch}')
        )
                  
        if validation_results['loss'] < best_validation_loss:
            best_validation_loss = validation_results['loss']
            wandb.log({'best_validation_loss': best_validation_loss})
            wandb.log({'reported_test_loss': test_results['loss']})
            wandb.log({'reported_test_results': test_results})


            
if __name__ == "__main__":
    train()

from functools import lru_cache
import timm
from torch import nn
import torch

    
class SolarIrradianceForecast(nn.Module):
    def __init__(self, hidden_layer_size=100, output_size=23, persistence=False):
        super(SolarIrradianceForecast, self).__init__()
        self.image_dims = 224
        self.hidden_layer_size = hidden_layer_size
        self.resize = nn.Upsample(self.image_dims)
        self.pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.output_size = output_size
        self.persistence = persistence
        self.fc = nn.Sequential(
            nn.Linear(768, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, self.output_size),
            nn.Tanh(),
        )

    def forward(self, images, persistence):
        batch_size = images.shape[0]
        resized_image = self.resize(images)
        # If three channel already, just run the model, else concatenate three times.
        if images.shape[1] == 3:
            features = self.pretrained_model.forward_features(resized_image)[:,-1,:]
        else:
            three_channel = torch.cat([resized_image, resized_image, resized_image], dim=1)
            features = self.pretrained_model.forward_features(three_channel)[:,-1,:]
        fc_output = self.fc(features)
        if self.persistence:
            output = fc_output + persistence
        else:
            output = fc_output
        return output
    
class VideoPlusReplication(nn.Module):
    def __init__(self, image_forecaster, replication_model):
        super(VideoPlusReplication, self).__init__()
        self.image_forecaster = image_forecaster
        self.replication_model = replication_model
        
    def forward(self, image, persistence=None):
        # Hard coding in time delay for now
        image = torch.cat([image, image, image], dim=1)
        forecasted_image = self.image_forecaster.forward(image, torch.stack([torch.Tensor([24])] * image.shape[0]).reshape(-1,1))
        forecasted_bands = self.replication_model.forward(forecasted_image)

        return forecasted_bands
    
    
class PretrainedWithImageForecaster(nn.Module):
    def __init__(self, image_forecaster, output_size=23, persistence=False, time_delay=24):
        super(PretrainedWithImageForecaster, self).__init__()
        self.image_forecaster = image_forecaster
        self.persistence = persistence
        self.output_size = output_size
        
        # This defines the part of the network that applied to the output of the
        # image forecast to provide an regression forecast
        self.forecast = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1,1,4,2),
            nn.Conv2d(1,1,4,2),
            
            nn.BatchNorm2d(1),
            nn.Conv2d(1,1,4,2),
            nn.Conv2d(1,1,4,2),
            
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(196,50),
            nn.Linear(50,50),
            nn.Linear(50,50),
            nn.Linear(50,output_size)
        )
        
    def forward(self, image, persistence):
        if image.shape[1] == 1:
            three_channel_image = torch.cat([image, image, image], dim=1)
        elif image.shape[1] == 3:
            three_channel_image = image
        
        # Hard coding in time delay for now
        forecasted_image = self.image_forecaster.forward(three_channel_image, torch.stack([torch.Tensor([24])] * image.shape[0]).reshape(-1,1))
        network_output = self.forecast(forecasted_image)
    
        if self.persistence:
            output = network_output + persistence
        else:
            output = network_output
        return output
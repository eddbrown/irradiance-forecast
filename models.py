from functools import lru_cache
import timm
from torch import nn
import torch

class IrradianceRegressor(nn.Module):
    def __init__(self, hidden_layer_size=100, output_size=23):
        super(IrradianceRegressor, self).__init__()
        self.image_dims = 224
        self.hidden_layer_size = hidden_layer_size
        self.resize = nn.Upsample(self.image_dims)
        self.pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(768, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, self.output_size),
        )

    def forward(self, images):
        batch_size = images.shape[0]
        resized_image = self.resize(images)
        # If three channel already, just run the model, else concatenate three times.
        if images.shape[1] == 3:
            features = self.pretrained_model.forward_features(resized_image)[:,-1,:]
        else:
            three_channel = torch.cat([resized_image, resized_image, resized_image], dim=1)
            features = self.pretrained_model.forward_features(three_channel)[:,-1,:]
        fc_output = self.fc(features)
        return fc_output

    
class IrradianceRegressorWithPersistence(nn.Module):
    def __init__(self, hidden_layer_size=100, output_size=23):
        super(IrradianceRegressorWithPersistence, self).__init__()
        self.image_dims = 224
        self.hidden_layer_size = hidden_layer_size
        self.resize = nn.Upsample(self.image_dims)
        self.pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(768+23, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, self.output_size),
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
        fc_output = self.fc(torch.cat((features, persistence),dim=1))
        return fc_output

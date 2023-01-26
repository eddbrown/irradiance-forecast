from functools import lru_cache
import timm
from torch import nn
import torch

class IrradianceRegressor(nn.Module):
    def __init__(self, hidden_layer_size=100, output_size=23):
        super(IrradianceRegressor, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.image_dims = 224
        self.resize = nn.Upsample(self.image_dims)
        self.pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(768, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, self.output_size),
            nn.LeakyReLU(),
        )
        self.positions = nn.Parameter(self.positional_encoding())
        
    def forward(self, images):
        batch_size = images.shape[0]
        resized_image = self.resize(images)
        position_encoding = torch.stack(batch_size * [self.positions])
        three_channel = torch.cat([positional_encoding, resized_image], dim=1)
        features = self.pretrained_model.forward_features(three_channel)[:,-1,:]
        fc_output = self.fc(features)
        return fc_output
    
    def positional_encoding(self):
        horizontal_positional_encoding = 1.0 * torch.ones((self.image_dims, self.image_dims))
        vertical_positional_encoding = 1.0 * torch.ones((self.image_dims, self.image_dims))
        for i in range(self.image_dims):
            horizontal_positional_encoding[:,i]=i/float(self.image_dims)
            vertical_positional_encoding[i,:]=i/float(self.image_dims)
        return torch.stack([horizontal_positional_encoding, vertical_positional_encoding])
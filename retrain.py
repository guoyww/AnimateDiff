import torch
import torch.nn as nn
from animatediff.models.unet_adjusted import UNet3DConditionModel
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load the original model's weights from the checkpoint
def load_original_weights(checkpoint_path):
    # Load the checkpoint directly as the original state dictionary
    original_state_dict = torch.load(checkpoint_path, map_location='cpu')
    return {key.replace("model.", ""): value for key, value in original_state_dict.items()}



# Identify and initialize new layers
def identify_and_initialize_new_layers(original_state_dict, new_model):
    new_state_dict = new_model.state_dict()
    
    # Identify new layers
    original_keys = set(original_state_dict.keys())
    new_keys = set(new_state_dict.keys())
    new_layers = new_keys - original_keys

    # Optionally: Initialize the new layers (example initialization provided)
    for layer_name in new_layers:
        if "running_mean" in layer_name or "running_var" in layer_name:
            # Skip BatchNorm internal states; they will be automatically updated during training
            continue
        elif "weight" in layer_name:
            layer_weight = dict(new_model.named_parameters())[layer_name]
            if isinstance(layer_weight, (nn.Parameter, torch.Tensor)) and len(layer_weight.shape) > 1:
                nn.init.kaiming_normal_(layer_weight, mode='fan_out', nonlinearity='relu')
        elif "bias" in layer_name:
            layer_bias = dict(new_model.named_parameters())[layer_name]
            if isinstance(layer_bias, (nn.Parameter, torch.Tensor)):
                nn.init.zeros_(layer_bias)





class LitUNet(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=1e-4):
        super(LitUNet, self).__init__()
        self.model = UNet3DConditionModel(in_channels, out_channels)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Assuming batch contains images and targets
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)  # Replace with your desired loss function
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def create_lit_model_with_weights(checkpoint_path, in_channels=3, out_channels=3):
    # Load original weights
    #original_weights = load_original_weights(checkpoint_path)

    # Instantiate the Lightning model
    #lit_model = LitUNet(in_channels=in_channels, out_channels=out_channels)
    
    # Transfer weights
    #lit_model.model = transfer_weights(original_weights, lit_model.model)

    # Identify and initialize new layers
    identify_and_initialize_new_layers(original_weights, lit_model.model)

    return lit_model

class RandomImageDataset(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random images and targets
        image = torch.randn(self.image_size)
        target = torch.randn(self.image_size)
        return image, target

# Create datasets and dataloaders
train_dataset = RandomImageDataset(num_samples=1000, image_size=(3, 512, 512))
val_dataset = RandomImageDataset(num_samples=200, image_size=(3, 512, 512))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

checkpoint_path = './models/Motion_Module/mm_sd_v15.ckpt'
lit_model = create_lit_model_with_weights(checkpoint_path)

trainer = pl.Trainer(max_epochs=5)  
trainer.fit(lit_model, train_dataloader, val_dataloader)  # This will start the training process




# Example usage
#new_model = UNetAdjusted(in_channels=3, out_channels=3)  # Adjust in_channels and out_channels as needed

# Load original weights
#original_weights = load_original_weights(checkpoint_path)
#print(torch.sum(new_model.up1.weight.data))

# Transfer weights
#new_model = transfer_weights(original_weights, new_model)

# Identify and initialize new layers
#identify_and_initialize_new_layers(original_weights, new_model)

#dummy_input = torch.randn((1, 3, 512, 512))  # Adjust the size and channels as needed
#output = new_model(dummy_input)
#print(output)
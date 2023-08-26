import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from animatediff.models.unet_adjusted import UNet3DConditionModel
from animatediff.data.dataset import WebVid10M
from safetensors import safe_open

def load_weights(model, checkpoint_path):
    # Load the checkpoint directly as the original state dictionary
    original_state_dict = torch.load(checkpoint_path, map_location='cpu')

    model_weights_state_dict = {key.replace("model.", ""): value for key, value in original_state_dict.items()}
    model.load_state_dict(model_weights_state_dict)

def transform_train_data(train_data):
    # Implement necessary transformations like normalization, scaling, etc. on the dataset 
    pass

def train_model(model, train_dataloader, optimizer, lr_scheduler, device):
    model.train()  # put model into training mode
    scaler = GradScaler()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # Forward pass
            with autocast():
                outputs = model(inputs)
                loss = F.mse_loss(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            if i % 50 == 0:  # Print Loss for every 50 iterations
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

        # Save the model after each epoch
        torch.save(model.state_dict(), f'./models/model_epoch_{epoch}.pth')

if __name__ == "__main__":
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'); 

    in_channels = 3
    out_channels = 3 
    # Instantiate the model and load the trained weights
    model = UNet3DConditionModel(in_channels, out_channels,unet_use_cross_frame_attention=False,unet_use_temporal_attention =True,motion_module_type="Vanilla")  
    model.to(device)
    load_weights(model, 'models/Motion_Module/mm_sd_v15.ckpt')
    identify_and_initialize_new_layers(original_weights, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # get the datasets
    train_data = WebVid10M(num_samples=1000, image_size=(3, 512, 512))
    train_dataset = transform_train_data(train_data)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
  
    train_model(model, train_dataloader, optimizer,lr_scheduler, device)  # Start Training
import torch
from hyper_model import COINModel
from hyper_dataloader import get_test_loader  # Assuming this function is defined to get your test data
from torchvision.utils import save_image
import os
import logging

# Load the model
model = COINModel()

# Load the state_dict from the checkpoint
checkpoint_path = "./checkpoint/state_dict=78999.ckpt"
state_dict = torch.load(checkpoint_path)

# Load the pre-trained weights into the model (strict=False to allow for any minor mismatches)
model.load_state_dict(state_dict, strict=False)

# Set the model to evaluation mode
model.eval()

# Get the test dataloader (assuming this returns (inputs, targets) in each batch)
test_loader = get_test_loader()

# Directory to save reconstructed images
ckpt_dir = './checkpoint/hyper'
log_file = os.path.join(ckpt_dir, 'training_log.csv')
output_dir = './reconstructed_images'
# os.makedirs(log_file, output_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_file, 'training.log'), level=logging.INFO)



# Loop over the test data and reconstruct images
for batch_idx, (inputs, targets) in enumerate(test_loader):
    with torch.no_grad():  # Disable gradient calculation during inference
        print("Input Shape:", inputs.shape)  # Input Shape: torch.Size([1, 3, 128, 128], (batch_size, channels, height, width) 형식으로 모델에 입력되고 있음
        print("Index:", batch_idx)
        outputs = model(inputs)  # Forward pass to get the reconstructed images
        print("Output Shape:", outputs.shape)  # Output Shape: torch.Size([1, 3])

        # 출력의 shape에서 값 추출
        if outputs.ndim == 2:
            height, width = 128, 128
            outputs = outputs.view(-1, 3, height, width)

        # Check the new shape
        print("Reshaped Output Shape:", outputs.shape)


    batch_size = 655361 
    # Loop through outputs if necessary
    for batch_idx in range(batch_size):
        # Use a specific index or identifier for the image
        image_id = batch_idx  # Assuming each output corresponds to the batch index
        print(f"Saving image with index: {image_id}")

        # Save the image using the correct identifier
        save_image(outputs, os.path.join(output_dir, f'reconstructed_image_{batch_idx}_{image_id}.png'))

print("Reconstruction complete. Check the 'reconstructed_images' directory for output.")

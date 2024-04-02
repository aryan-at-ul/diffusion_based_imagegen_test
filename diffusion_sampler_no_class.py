import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)



#classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


transform = transforms.Compose([
    transforms.Pad(2),  # Add padding of 2 on each side to increase image size from 28x28 to 32x32, error in attention mlp for channel 1?????
    transforms.ToTensor()
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

import torch
import torch.optim as optim
from tqdm import tqdm
from ddpm_proteins import Unet, GaussianDiffusion

# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels = 1

)

diffusion = GaussianDiffusion(
    model.to(device),
    image_size =32,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)


diffusion = diffusion.to(device)

# Initialize the optimizer (e.g., Adam, SGD, etc.)
optimizer = optim.Adam(diffusion.parameters(), lr=0.0001)


num_epochs = 1000
best_loss = float('inf')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = diffusion(inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Check and print loss every 300 mini-batches
        if i % 300 == 299:
            avg_loss = running_loss / 300
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
            # Update the best loss and save the model if this is the lowest loss so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save the model
                torch.save(model.state_dict(), 'best_model.pth')
            running_loss = 0.0

print('Training done')


import matplotlib.pyplot as plt
import numpy as np

# Flag to check if the dataset is MNIST
dataset = "mnist"

sampled_images = diffusion.sample(batch_size=4)
print(sampled_images.shape) # Expected to be (4, 3, 32, 32)
sampled_images = sampled_images.detach().cpu()

def tensor_to_image(tensor, is_mnist=False):
    if is_mnist:
        # Convert (3, 32, 32) back to (1, 28, 28)
        tensor = tensor.mean(dim=0, keepdim=True)  # Convert to grayscale by averaging channels
        tensor = tensor[0, 2:-2, 2:-2]  # Crop the image to 28x28
        tensor = tensor.unsqueeze(0)  # Add a channel dimension
    tensor = tensor.numpy()
    tensor = np.transpose(tensor, (1, 2, 0))
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

for i, ax in enumerate(axes.flat):
    img = tensor_to_image(sampled_images[i], is_mnist=(dataset == "mnist"))
    ax.imshow(img.squeeze(), cmap='gray' if dataset == "mnist" else None)
    ax.axis('off')

plt.show()






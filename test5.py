import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the pre-trained DINO model (vits8)
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=True)
vits8.eval()  # Set the model to evaluation mode

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_loader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True)

# Define a simple classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

# Define the dimensions of the feature embeddings
embedding_size = 384

# Initialize the classifier
classifier = SimpleClassifier(input_size=embedding_size, num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

start_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in cifar10_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract feature embeddings using DINO model
        with torch.no_grad():
            features = vits8(images)

        # Flatten the features
        features = features.view(features.size(0), -1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = classifier(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(cifar10_loader):.4f}")

print(f"Training finished in {(time.time() - start_time):.2f} seconds")

# Visualize the input vs test results
num_samples = 10
images, labels = next(iter(cifar10_loader))
images = images[:num_samples].to(device)
labels = labels[:num_samples].numpy()

# Extract feature embeddings using DINO model
with torch.no_grad():
    features = vits8(images)

# Flatten the features
features = features.view(features.size(0), -1)

# Forward pass through the classifier
outputs = classifier(features)
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().numpy()

# Plot the input images and their corresponding predicted labels
fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
for i in range(num_samples):
    image = images[i].cpu().numpy().transpose((1, 2, 0))
    image = (image + 1) / 2  # Unnormalize
    axes[i].imshow(image)
    axes[i].set_title(f"Pred: {predicted[i]}, True: {labels[i]}")
    axes[i].axis('off')

plt.show()

# Training finished, visualize the input vs test results
num_samples = 10
images, labels = next(iter(cifar10_loader))
images = images[:num_samples].to(device)
labels = labels[:num_samples].numpy()

# Extract feature embeddings using DINO model
with torch.no_grad():
    features = vits8(images)

# Flatten the features
features = features.view(features.size(0), -1)

# Forward pass through the classifier
outputs = classifier(features)
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().numpy()

# Plot the input images and their corresponding predicted labels
fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))  # Increase figure width
for i in range(num_samples):
    image = images[i].cpu().numpy().transpose((1, 2, 0))
    image = (image + 1) / 2  # Unnormalize
    axes[i].imshow(image)
    axes[i].set_title(f"Pred: {predicted[i]}\nTrue: {labels[i]}")  # Use '\n' for newline
    axes[i].axis('off')

plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()


# Compute and visualize the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Iterate through the dataset to obtain true and predicted labels
for images, labels in cifar10_loader:
    images = images.to(device)

    # Extract feature embeddings using DINO model
    with torch.no_grad():
        features = vits8(images)

    # Flatten the features
    features = features.view(features.size(0), -1)

    # Forward pass through the classifier
    outputs = classifier(features)
    _, predicted = torch.max(outputs, 1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted.cpu().numpy())

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:11:53 2024

@author: aavyasrivastava
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

## Define parameters

#Typically ranges from 0.0001 to 0.1. 
#Smaller values lead to slower but more precise convergence, while larger values 
#may cause the model to overshoot the minimum.
learning_rate = 0.001

#Commonly ranges from 8 to 256.
#Smaller batch sizes provide a noisier estimate of the gradient but can lead
#to faster convergence and better generalization. Larger  sizes provide a 
#smoother estimate of the gradient but  require more memory and computational resources.
batch_size = 256

#Usually ranges from 10 to 1000 or more. The number of epochs determines 
#how many times the entire training dataset is passed forward and backward
# through the CNN. It depends on factors such as convergence speed, dataset 
#complexity, and computational resources.
num_epochs = 100

#Typically ranges from 0.1 to 0.5. Dropout is a regularization technique used 
#to prevent overfitting by randomly setting a fraction of input units to zero 
#during training. A dropout rate of 0.5 means that half of the units will be 
#dropped out during training. Smaller values may not provide sufficient regularization, 
#while larger values may lead to underfitting.
dropout_rate = 0.5

# Define CNN architecture
class SkinCancerCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):  # Add dropout_rate as an argument
        super(SkinCancerCNN, self).__init__()
        # Define convolutional layers, pooling layers, and fully connected layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        self.fc2 = nn.Linear(128, 2)  # 2 output classes (benign, malignant)

    def forward(self, x):
        # Forward pass through convolutional layers and fully connected layers
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = self.fc2(x)
        return x


# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='/Users/aavyasrivastava/train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initializing model, loss function, and optimizer
model = SkinCancerCNN(dropout_rate=dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and test losses
train_losses = []
test_losses = []

# Load the test dataset
test_dataset = ImageFolder(root='/Users/aavyasrivastava/test/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    # Compute training loss after each epoch
    train_losses.append(running_loss / len(train_loader))

    # Evaluate model on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
    
    # Compute average test loss
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    model.train()

print('Finished Training')

#loss function graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'skin_cancer_detection_model.pth')
#Final model parameters: Learning rate: 0.001, Batch size: 128, 
#Number of epochs: 50, Dropout rate: 0.5

#TEST
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Define the same transform as used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the trained model
model = SkinCancerCNN()
model.load_state_dict(torch.load('skin_cancer_detection_model.pth'))
model.eval()  # Set model to evaluation mode

# Load the different test dataset -- 32 images
test_dataset = ImageFolder(root='/Users/aavyasrivastava/test1/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define variables for tracking accuracy
correct = 0
total = 0
predicted_labels = []
true_labels = []
benign_count = 0
malignant_count = 0

# Iterate over test dataset
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        
        # Convert predicted labels to class names
        predicted_classes = ['benign' if pred == 0 else 'malignant' for pred in predicted]
        
        # Convert true labels to class names
        true_classes = ['benign' if label == 0 else 'malignant' for label in labels]
        
        # Extend lists for true and predicted labels
        predicted_labels.extend(predicted_classes)
        true_labels.extend(true_classes)
        
        # Count number of benign and malignant images
        benign_count += (predicted == 0).sum().item()
        malignant_count += (predicted == 1).sum().item()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Encode class names into numeric labels
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)
predicted_labels_encoded = label_encoder.transform(predicted_labels)

# Calculate performance metrics
accuracy = accuracy_score(true_labels_encoded, predicted_labels_encoded)
precision = precision_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
recall = recall_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
f1 = f1_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
conf_matrix = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
roc_auc = roc_auc_score(true_labels_encoded, predicted_labels_encoded)

print('Number of benign images:', benign_count)
print('Number of malignant images:', malignant_count)
print('Accuracy of the network on the test images: %.2f %%' % (100 * accuracy))
print('Precision of the network on the test images:', precision)
print('Recall of the network on the test images:', recall)
print('F1 Score of the network on the test images:', f1)
print('Confusion Matrix of the network on the test images:')

#Shows type 1 and type 2 error
print(conf_matrix) # TP FN
                    #FP TN
                    
print('ROC AUC Score of the network on the test images:', roc_auc)
print('Number Correct: ', correct, ' out of ', total)

#Benign or malignant with image names
import os
from PIL import Image
from torchvision.datasets import ImageFolder

# Load the test dataset -- 32 images
test_folder = '/Users/aavyasrivastava/test1/'
test_dataset = ImageFolder(root=test_folder, transform=transform)

# Iterate over test dataset
for image_path, label in test_dataset.imgs:
    image_name = os.path.basename(image_path)
    
    # Load and transform the image
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Perform prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.softmax(output, dim=1)
        predicted_class = 'benign' if probability[0][0] > probability[0][1] else 'malignant'
    
    # Print prediction with image name
    print(f"{image_name}: {predicted_class}")

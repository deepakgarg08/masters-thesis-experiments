import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset for UTKFACE
class UTKFaceDataset(Dataset):
  def __init__(self, img_dir, transform=None):
      self.img_dir = img_dir
      self.transform = transform
      self.img_labels = []

      for img in os.listdir(img_dir):
          parts = img.split('_')
          if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
              age = int(parts[0])
              gender = int(parts[1])
              if gender in [0, 1]:  # Ensure gender is either 0 or 1
                  self.img_labels.append((img, age, gender))
              else:
                  print(f"Skipping file with invalid gender label: {img}")
          else:
              print(f"Skipping file with unexpected format: {img}")

  def __len__(self):
      return len(self.img_labels)

  def __getitem__(self, idx):
      img_name, age, gender = self.img_labels[idx]
      img_path = os.path.join(self.img_dir, img_name)
      image = Image.open(img_path).convert('RGB')
      if self.transform:
          image = self.transform(image)
      return image, age, gender

# Define the model
class FaceRecognitionModel(nn.Module):
  def __init__(self):
      super(FaceRecognitionModel, self).__init__()
      self.features = models.resnet50(pretrained=True)
      self.features.fc = nn.Identity()  # Remove the last layer
      self.dropout = nn.Dropout(0.5)
      self.age_classifier = nn.Linear(2048, 1)
      self.gender_classifier = nn.Linear(2048, 2)

  def forward(self, x):
      x = self.features(x)
      x = self.dropout(x)
      age = self.age_classifier(x)
      gender = self.gender_classifier(x)
      return age, gender

def train_and_evaluate():
  # Data transformations with advanced augmentation
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
  ])

  # Load dataset
  dataset_path = 'UTKFACE-ALL'
  dataset = UTKFaceDataset(img_dir=dataset_path, transform=transform)
  train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
  train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

  # Initialize model, loss functions, and optimizer
  model = FaceRecognitionModel().to(device)  # Move model to GPU if available
  criterion_age = nn.MSELoss()
  criterion_gender = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Added L2 regularization
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

  # Load the best model weights if available
  if os.path.exists('best_model.pth'):
      model.load_state_dict(torch.load('best_model.pth'))
      print("Loaded model state from best_model.pth.")

  # Early stopping parameters
  patience = 5
  best_loss = np.inf
  patience_counter = 0

  # Directory to save model checkpoints
  checkpoint_dir = 'model_checkpoints'
  os.makedirs(checkpoint_dir, exist_ok=True)

  training_logs_dir = "training_logs"
  os.makedirs(training_logs_dir, exist_ok=True)

  # Log file to save training and validation metrics
  log_file_path = os.path.join(training_logs_dir, 'training_logs.txt')

  with open(log_file_path, 'w') as log_file:
      # Write headers for the log file
      log_file.write("Epoch, Train Loss Age, Val Loss Age, Train Loss Gender, Val Loss Gender, "
                     "Train Accuracy Gender, Val Accuracy Gender, Train MAE Age, Val MAE Age\n")
      # Training loop
      num_epochs = 50
      train_losses_age, val_losses_age = [], []
      train_losses_gender, val_losses_gender = [], []
      train_accuracies_gender, val_accuracies_gender = [], []
      train_mae_age, val_mae_age = [], []

      for epoch in range(num_epochs):
          model.train()
          running_loss_age = 0.0
          running_loss_gender = 0.0
          correct_gender = 0
          total_gender = 0
          total_age_error = 0

          for images, ages, genders in train_loader:
              images, ages, genders = images.to(device), ages.to(device), genders.to(device)  # Move data to GPU

              optimizer.zero_grad()
              outputs_age, outputs_gender = model(images)
              loss_age = criterion_age(outputs_age.squeeze(), ages.float())
              loss_gender = criterion_gender(outputs_gender, genders)
              loss = loss_age + loss_gender
              loss.backward()
              optimizer.step()

              running_loss_age += loss_age.item()
              running_loss_gender += loss_gender.item()
              _, predicted_gender = torch.max(outputs_gender, 1)
              correct_gender += (predicted_gender == genders).sum().item()
              total_gender += genders.size(0)
              total_age_error += mean_absolute_error(ages.cpu().numpy(), outputs_age.squeeze().detach().cpu().numpy()) * ages.size(0)

          train_losses_age.append(running_loss_age / len(train_loader))
          train_losses_gender.append(running_loss_gender / len(train_loader))
          train_accuracies_gender.append(correct_gender / total_gender)
          train_mae_age.append(total_age_error / total_gender)

          # Validation
          model.eval()
          val_loss_age = 0.0
          val_loss_gender = 0.0
          correct_gender = 0
          total_gender = 0
          total_age_error = 0

          with torch.no_grad():
              for images, ages, genders in test_loader:
                  images, ages, genders = images.to(device), ages.to(device), genders.to(device)  # Move data to GPU
                  outputs_age, outputs_gender = model(images)
                  loss_age = criterion_age(outputs_age.squeeze(), ages.float())
                  loss_gender = criterion_gender(outputs_gender, genders)
                  val_loss_age += loss_age.item()
                  val_loss_gender += loss_gender.item()
                  _, predicted_gender = torch.max(outputs_gender, 1)
                  correct_gender += (predicted_gender == genders).sum().item()
                  total_gender += genders.size(0)
                  total_age_error += mean_absolute_error(ages.cpu().numpy(), outputs_age.squeeze().cpu().numpy()) * ages.size(0)

          val_losses_age.append(val_loss_age / len(test_loader))
          val_losses_gender.append(val_loss_gender / len(test_loader))
          val_accuracies_gender.append(correct_gender / total_gender)
          val_mae_age.append(total_age_error / total_gender)
          scheduler.step(val_losses_age[-1] + val_losses_gender[-1])

          print(f'Epoch {epoch+1}, Train Loss Age: {train_losses_age[-1]}, Val Loss Age: {val_losses_age[-1]}')
          print(f'Epoch {epoch+1}, Train Loss Gender: {train_losses_gender[-1]}, Val Loss Gender: {val_losses_gender[-1]}')
          print(f'Epoch {epoch+1}, Train Accuracy Gender: {train_accuracies_gender[-1]}, Val Accuracy Gender: {val_accuracies_gender[-1]}')
          print(f'Epoch {epoch+1}, Train MAE Age: {train_mae_age[-1]}, Val MAE Age: {val_mae_age[-1]}')

          # Save model state at the end of each epoch
          model_checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
          torch.save(model.state_dict(), model_checkpoint_path)
          print(f'Model saved to {model_checkpoint_path}')

          # Write epoch results to the log file
          log_file.write(f"{epoch+1}, {train_losses_age[-1]}, {val_losses_age[-1]}, "
                         f"{train_losses_gender[-1]}, {val_losses_gender[-1]}, "
                         f"{train_accuracies_gender[-1]}, {val_accuracies_gender[-1]}, "
                         f"{train_mae_age[-1]}, {val_mae_age[-1]}\n")
          # Early stopping
          if val_losses_age[-1] + val_losses_gender[-1] < best_loss:
              best_loss = val_losses_age[-1] + val_losses_gender[-1]
              patience_counter = 0
              torch.save(model.state_dict(), 'best_model.pth')
          else:
              patience_counter += 1
              if patience_counter >= patience:
                  print("Early stopping")
                  break

  # Load the best model for final evaluation
  model.load_state_dict(torch.load('best_model.pth'))

  # Create a figure with specified size
  fig, axes = plt.subplots(1, 3, figsize=(20, 8))

  # Plot training vs validation accuracy for gender and MAE for age
  axes[0].plot(train_accuracies_gender, label='Train_Gender_Acc', color='blue')
  axes[0].plot(val_accuracies_gender, label='Val_Gender_Acc', color='lightgreen')
  axes[0].plot(train_mae_age, label='Train_Age_Acc', color='red')
  axes[0].plot(val_mae_age, label='Val_Age_Acc', color='green')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Accuracy')
  axes[0].set_title('Gender and Age Prediction Accuracy')
  axes[0].legend()

  # Plot training vs validation loss for age
  axes[1].plot(train_losses_age, label='Train Age Loss ', color='blue')
  axes[1].plot(val_losses_age, label='Val Age Loss', color='green')
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Loss')
  axes[1].set_title('Age Prediction Loss')
  axes[1].legend()

  # Plot training vs validation loss for gender
  axes[2].plot(train_losses_gender, label='Train Gender Loss', color='blue')
  axes[2].plot(val_losses_gender, label='Val Gender Loss', color='green')
  axes[2].set_xlabel('Epoch')
  axes[2].set_ylabel('Loss')
  axes[2].set_title('Gender Prediction Loss')
  axes[2].legend()

  # Show the plots
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  train_and_evaluate()
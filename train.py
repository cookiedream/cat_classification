import torch
from dataloader import train_loader, val_loader
import yaml
from Model.main import model_dict, print_model_parameters
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchsummary import summary

with open('train.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_class = model_dict[config['TRAINING']['MODEL']]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model_class(n_class=config['TRAINING']['N_CLASS']).to(device)


summary(model, input_size=(3, 224, 224))  # 以您模型的輸入大小替換 (3, 224, 224)

loss_name = config['TRAINING']['LOSS']
if loss_name == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
elif loss_name == 'MSELoss':
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss function: {loss_name}")

optimizer = optim.SGD(model.parameters(),
                      lr=config['TRAINING']['LEARNING_RATE'])

# Initialize TensorBoard
writer = SummaryWriter()

best_val_acc = 0.0
if not os.path.exists('weights'):
    os.makedirs('weights')

# Calculate and print the total number of parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

for epoch in range(config['TRAINING']['EPOCHS']):
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['TRAINING']['EPOCHS']} Training",
                     bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_bar:

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)

    model.eval()
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['TRAINING']['EPOCHS']} Validating",
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    for val_images, val_labels in val_bar:
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        val_outputs = model(val_images)
        loss = criterion(val_outputs, val_labels)
        val_loss += loss.item()

        _, val_predicted = torch.max(val_outputs.data, 1)
        val_total += val_labels.size(0)
        val_correct += (val_predicted == val_labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    writer.add_scalar('Loss/Val', val_loss, epoch)
    writer.add_scalar('Accuracy/Val', val_acc, epoch)

    print(f"Epoch {epoch+1} Completed: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if not os.path.exists('weights'):
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'weights/{model}_best_weights.pt')
            print("Saved new best weights.")

writer.close()

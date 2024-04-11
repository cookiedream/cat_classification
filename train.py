import torch
from dataloader import train_loader, val_loader
import yaml
from Model.main import model_dict
import torch.nn as nn
import torch.optim as optim


with open('train.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


model_class = model_dict[config['TRAINING']['MODEL']]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define your model
model = model_class(
    n_class=config['TRAINING']['N_CLASS']).to(device)

# Define loss function and optimizer
loss_name = config['TRAINING']['LOSS']
if loss_name == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
elif loss_name == 'MSELoss':
    criterion = nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss function: {loss_name}")

optimizer = optim.SGD(model.parameters(),
                      lr=config['TRAINING']['LEARNING_RATE'])

# Training loop
for epoch in range(config['TRAINING']['EPOCHS']):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    # Compute validation accuracy and loss
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()

            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    # Print statistics
    print(f"Epoch {epoch+1}/{config['TRAINING']['EPOCHS']}:")
    print(
        f"Train Loss: {running_loss / len(train_loader):.4f} | Train Acc: {100 * correct / total:.2f}%")
    print(
        f"Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {100 * val_correct / val_total:.2f}%")

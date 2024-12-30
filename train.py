import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from dataset import get_dataloader
import time
import logging


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            boxes = batch['boxes'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, boxes, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print(f'Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                boxes = batch['boxes'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(images)
                loss = criterion(outputs, boxes, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{Config.MODEL_SAVE_PATH}/best_model.pth')

    return model


def train(model_type):
    # Setup logging
    logging.basicConfig(filename=f'{Config.LOGS_PATH}/training.log', level=logging.INFO)

    # Get dataloaders
    train_loader = get_dataloader(Config.DATASET_PATH, Config.BATCH_SIZE, train=True)
    val_loader = get_dataloader(Config.DATASET_PATH, Config.BATCH_SIZE, train=False)

    # Initialize model, criterion, and optimizer
    model = get_model(model_type)
    criterion = torch.nn.SmoothL1Loss()  # For bounding box regression
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train the model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        Config.NUM_EPOCHS
    )

    return trained_model
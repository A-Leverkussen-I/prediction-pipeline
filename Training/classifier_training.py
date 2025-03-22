import os
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from sklearn.model_selection import train_test_split

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Soccer Player Pose Classification Training Script"
    )
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help="Path to dataset folder organized by class subdirectories")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0010084101984587203,
                        help="Initial learning rate")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument('--dropout_rate', type=float, default=0.5018474396618718,
                        help="Dropout rate for the classifier head")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience for early stopping (number of epochs with no improvement)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory for saving checkpoints, logs, and evaluation results")
    args = parser.parse_args()
    return args

def get_data_transforms():
    """
    Define data augmentation and normalization transforms for training, validation, and testing.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15,translate=(0.1,0.1), scale=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def get_dataloaders(data_dir, batch_size, num_workers):
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    """
    train_transforms, val_test_transforms = get_data_transforms()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    targets = full_dataset.targets
    indices = np.arange(len(full_dataset))
    
    # Stratified split for train, validation, and test
    train_idx, temp_idx, _, _ = train_test_split(indices, targets, test_size=0.3, stratify=targets, random_state=42)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx, _, _ = train_test_split(temp_idx, temp_targets, test_size=0.5, stratify=temp_targets, random_state=42)
    
    def get_split_dataset(split_indices, transform):
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataset.samples = [dataset.samples[i] for i in split_indices]
        dataset.imgs = dataset.samples  # For backward compatibility
        dataset.targets = [dataset.targets[i] for i in split_indices]
        return dataset
    
    train_dataset = get_split_dataset(train_idx, train_transforms)
    val_dataset = get_split_dataset(val_idx, val_test_transforms)
    test_dataset = get_split_dataset(test_idx, val_test_transforms)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    return dataloaders, full_dataset.classes

def build_model(num_classes, dropout_rate=0.5):
    """
    Build the transfer learning model using a pre-trained ResNet50.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze all layers except layer4 and the fully connected layer
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    return model

def train_model(model, dataloaders, criterion, optimizer, device,
                num_epochs=25, patience=5, output_dir='outputs', writer=None):
    """
    Train the model with early stopping.
    
    This version only saves the final best model checkpoint to reduce disk usage.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            if writer:
                writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)
            
            # Update best model based on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete.")
    model.load_state_dict(best_model_wts)
    
    # Save only the final best model checkpoint
    final_checkpoint_path = os.path.join(output_dir, "final_best_model.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model checkpoint saved: {final_checkpoint_path}")
    
    return model

def evaluate_model(model, dataloader, device, class_names, output_dir='outputs'):
    """
    Evaluate the model on the test set and compute metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Test Accuracy: {overall_acc:.4f}")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )
    
    print("Per-Class Metrics:")
    for idx, classname in enumerate(class_names):
        print(f"  {classname:20s} -> Precision: {precision[idx]:.4f}  Recall: {recall[idx]:.4f}  F1-Score: {f1[idx]:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save classification report
    cls_report = classification_report(all_labels, all_preds, target_names=class_names)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(cls_report)
    print(f"Classification report saved to {report_path}")
    
    return overall_acc, precision, recall, f1

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloaders, class_names = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(class_names), dropout_rate=args.dropout_rate)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.learning_rate)
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Train the model with early stopping; only the best model is saved
    model = train_model(model, dataloaders, criterion, optimizer, device,
                        num_epochs=args.epochs, patience=args.patience,
                        output_dir=args.output_dir, writer=writer)
    
    # Evaluate the model on the test set
    overall_acc, _, _, _ = evaluate_model(model, dataloaders['test'], device, class_names, output_dir=args.output_dir)
    
    # Write the evaluation metric to avg_accuracy.txt for hyperparameter tuning
    metric_file = os.path.join(args.output_dir, "avg_accuracy.txt")
    try:
        with open(metric_file, "w") as f:
            f.write(f"{overall_acc:.4f}")
        print(f"Average accuracy metric saved to {metric_file}")
    except Exception as e:
        print(f"Error saving metric file: {e}")
    
    writer.close()

if __name__ == "__main__":
    main()

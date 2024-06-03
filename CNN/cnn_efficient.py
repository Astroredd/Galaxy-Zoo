from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn as nn
import torch
import timm
import numpy as np
import copy
import os
import tqdm


# Función para agregar ruido gaussiano
def add_gaussian_noise(image, mean=0, std=0.1):
    image = np.array(image) / 255.0
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

# Función para agregar ruido de Poisson
def add_poisson_noise(image):
    image = np.array(image) / 255.0
    noisy_image = np.random.poisson(image * 255.0) / 255.0
    noisy_image = np.clip(noisy_image, 0, 1)
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

# Función para agregar ambos ruidos: Gaussiano + Poisson
def add_combined_noise(image):
    image = np.array(image) / 255.0
    gaussian_noise = np.random.normal(0, 0.1, image.shape)
    poisson_noise = np.random.poisson(image * 255.0) / 255.0
    combined_image = image + gaussian_noise + poisson_noise
    combined_image = np.clip(combined_image, 0, 1)
    return Image.fromarray((combined_image * 255).astype(np.uint8))

# Función para no aplicar transformaciones
def no_transform(image):
    return image

# Función para aplicar una transformación de ruido aleatoria o ninguna
def apply_random_noise(image):
    noise_functions = [add_gaussian_noise, add_poisson_noise, add_combined_noise, no_transform]
    noise_function = np.random.choice(noise_functions)
    return noise_function(image)

# Dataset personalizado
class GalaxyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = str(int(self.dataframe.iloc[idx]["GalaxyID"])) + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = self.dataframe.iloc[idx][1:].values.astype(np.float32)
        return image, torch.tensor(labels)

# Cargar los archivos CSV
df_train = pd.read_csv('training_solutions_rev1.csv')

# Dividir los datos de entrenamiento en entrenamiento y validación
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

# Especificar el directorio correcto de imágenes
train_img_dir = 'images_training_rev1'

# Transformaciones de aumento de datos y normalización
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño de entrada para EfficientNet
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.Lambda(apply_random_noise),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Tamaño de entrada para EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear DataLoader para entrenamiento y validación
train_dataset = GalaxyDataset(dataframe=train_df, img_dir=train_img_dir, transform=transform_train)
val_dataset = GalaxyDataset(dataframe=val_df, img_dir=train_img_dir, transform=transform_val_test)

# Función para entrenar el modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, scheduler=None, patience = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()
    all_labels = []
    all_preds = []
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0

            for inputs, labels in tqdm(dataloader, desc=f"{phase} Epoch {epoch}/{num_epochs - 1}", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)

                if phase == 'val':
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(outputs.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            if phase == 'train':
                epoch_train_loss = epoch_loss
                train_losses.append(epoch_loss)
            else:
                epoch_val_loss = epoch_loss
                val_losses.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if scheduler:
            scheduler.step()

        print()

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model, all_labels, all_preds, train_losses, val_losses

if __name__ == '__main__':
    # Configuraciones de CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Asegúrate de que el dispositivo está siendo utilizado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Monitorización de la memoria
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

    # Parámetros seleccionados
    lr = 1e-4
    batch_size = 24
    num_epochs = 40
    patience = 5  # Número de épocas sin mejora antes de detener el entrenamiento

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    # Crear el modelo EfficientNet
    efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True)
    num_ftrs = efficientnet_model.classifier.in_features
    efficientnet_model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, len(df_train.columns) - 1),
        nn.Sigmoid()
    )
    efficientnet_model = efficientnet_model.to(device)
    
    optimizer = torch.optim.Adam(efficientnet_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0004)
    
    trained_model, all_labels, all_preds, train_losses, val_losses = train_model(
        efficientnet_model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler, patience=patience
    )

    val_loss = 0.0
    efficientnet_model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')

    # Guardar el modelo
    torch.save(trained_model.state_dict(), 'best_efficientnet_model_efficient.pth')
    
    # Guardar las etiquetas y predicciones para la matriz de confusión
    np.save('all_labels_efficiet.npy', all_labels)
    np.save('all_preds_efficient.npy', all_preds)

    # Guardar las pérdidas de entrenamiento y validación
    np.save('train_losses_efficient.npy', train_losses)
    np.save('val_losses_efficient.npy', val_losses)

    print('Modelo y métricas guardados correctamente.')

    # Plotear la curva de entrenamiento
    import matplotlib.pyplot as plt

    # Cargar las pérdidas guardadas
    train_losses = np.load('train_losses_efficient.npy')
    val_losses = np.load('val_losses_efficient.npy')

    # Plotear la curva de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

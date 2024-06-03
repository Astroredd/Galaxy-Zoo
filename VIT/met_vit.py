import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm

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

if __name__ == '__main__':
    # Cargar los archivos CSV
    df_train = pd.read_csv('training_solutions_rev1.csv')

    # Especificar el directorio correcto de imágenes
    train_img_dir = 'images_training_rev1'

    # Transformaciones de validación
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Tamaño de entrada para DeiT-tiny
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_df = df_train.sample(frac=0.2, random_state=42)  # Obtener un subconjunto de validación
    val_dataset = GalaxyDataset(dataframe=val_df, img_dir=train_img_dir, transform=transform_val_test)

    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Cargar el modelo guardado
    deit_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    num_ftrs = deit_model.head.in_features
    deit_model.head = nn.Sequential(
        nn.Linear(num_ftrs, len(df_train.columns) - 1),
        nn.Sigmoid()
    )

    # Cargar el estado del modelo guardado
    state_dict = torch.load('best_deit_model.pth')

    # Crear un nuevo state_dict solo con las claves que coinciden
    model_state_dict = deit_model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

    # Actualizar el state_dict del modelo con las capas coincidentes
    model_state_dict.update(new_state_dict)
    deit_model.load_state_dict(model_state_dict)

    deit_model = deit_model.to(device)

    # Cargar las predicciones y etiquetas guardadas
    all_labels = np.load('all_labels_deit.npy')
    all_preds = np.load('all_preds_deit.npy')

    # Calcular la pérdida de validación
    criterion = nn.MSELoss()
    val_loss = criterion(torch.tensor(all_preds), torch.tensor(all_labels)).item()
    print(f'Validation Loss: {val_loss:.4f}')

    # Calcular la precisión (Accuracy)
    # Si las predicciones son probabilidades, conviértalas en etiquetas de clase (para clasificación multiclase)
    all_preds_classes = np.argmax(all_preds, axis=1)
    all_labels_classes = np.argmax(all_labels, axis=1)
    accuracy = accuracy_score(all_labels_classes, all_preds_classes)
    print(f'Accuracy: {accuracy:.4f}')


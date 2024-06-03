import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score
import pandas as pd
from efficientnet_pytorch import EfficientNet

# Cargar los archivos CSV
df_train = pd.read_csv('training_solutions_rev1.csv')

# Cargar el modelo guardado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(df_train.columns) - 1)

# Cargar el estado del modelo guardado
state_dict = torch.load('best_efficientnet_model_efficient.pth')

# Crear un nuevo state_dict solo con las claves que coinciden
model_state_dict = model.state_dict()
new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

# Actualizar el state_dict del modelo con las capas coincidentes
model_state_dict.update(new_state_dict)
model.load_state_dict(model_state_dict)

model = model.to(device)

# Cargar las predicciones y etiquetas guardadas
all_labels = np.load('all_labels_efficiet.npy')
all_preds = np.load('all_preds_efficient.npy')

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


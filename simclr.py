import os
import glob
import tarfile
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

# =====================================================================
# 1. ARCHITECTURES (ENCODEUR ET DÉCODEUR)
# =====================================================================

class SimCLR_Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLR_Encoder, self).__init__()
        resnet = models.resnet18(weights=None)
        # Modification pour 4 canaux (T1, T1ce, T2, Flair)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) # On garde les features spatiales
        
        # Pour SimCLR, on aplatit et on projette
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x) # Shape: (B, 512, H', W')
        if return_features:
            return features
            
        h = self.pool(features)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return h, z

class SegmentationModel(nn.Module):
    def __init__(self, encoder):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        
        # Décodeur simple (Upsampling) pour revenir à la taille de l'image d'origine (240x240)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Couche finale pour obtenir 1 seul canal (le masque binaire)
            nn.Conv2d(16, 1, kernel_size=3, padding=1) 
            # Note: On n'utilise pas Sigmoid ici car on va utiliser BCEWithLogitsLoss
        )

    def forward(self, x):
        features = self.encoder(x, return_features=True)
        # On interpole si la taille finale ne matche pas exactement 240x240 à cause des arrondis
        out = self.decoder(features)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-1e4)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z.device)
        return self.criterion(sim_matrix, labels)

# =====================================================================
# 2. DATASETS (SimCLR et Segmentation)
# =====================================================================

# --- Pipeline d'augmentation pour SimCLR ---
augmentation_pipeline = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
])
def add_gaussian_noise(tensor, mean=0., std=0.05):
    return tensor + (torch.randn(tensor.size()) * std + mean)

# --- Dataset SimCLR ---
class BraTS_SimCLR_Dataset(Dataset):
    def __init__(self, data_dir):
        self.patient_folders = glob.glob(os.path.join(data_dir, "BraTS2021_*"))
    def __len__(self): return len(self.patient_folders)
    def __getitem__(self, idx):
        patient_dir = self.patient_folders[idx]
        patient_id = os.path.basename(patient_dir)
        slice_idx = 75
        
        # Chargement rapide des 4 modalités
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [nib.load(os.path.join(patient_dir, f"{patient_id}_{m}.nii.gz")).get_fdata()[:, :, slice_idx] for m in modalities]
        
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min()) if img.max() > 0 else img
        
        imgs = list(map(normalize, imgs))
        base_image = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)

        view_a = add_gaussian_noise(augmentation_pipeline(base_image))
        view_b = add_gaussian_noise(augmentation_pipeline(base_image))
        return view_a, view_b

# --- Dataset Segmentation ---
class BraTS2DDataset(Dataset):
    def __init__(self, data_dir):
        self.patient_folders = glob.glob(os.path.join(data_dir, "BraTS2021_*"))
        print(f"Dataset Segmentation : Trouvé {len(self.patient_folders)} patients.")

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_dir = self.patient_folders[idx]
        patient_id = os.path.basename(patient_dir)
        slice_idx = 75

        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [nib.load(os.path.join(patient_dir, f"{patient_id}_{m}.nii.gz")).get_fdata()[:, :, slice_idx] for m in modalities]
        mask = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")).get_fdata()[:, :, slice_idx]

        def normalize(img):
            return (img - img.min()) / (img.max() - img.min()) if img.max() > 0 else img

        imgs = list(map(normalize, imgs))
        image = np.stack(imgs, axis=0)

        # Binarisation du masque
        mask[mask > 0] = 1
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# =====================================================================
# 3. FONCTIONS D'ENTRAÎNEMENT
# =====================================================================

def train_simclr(model, dataloader, device, epochs, save_path):
    print("\n--- DÉBUT DE LA PHASE 1 : PRÉ-ENTRAÎNEMENT SIMCLR ---")
    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    historique_loss = []
    accumulation_steps = 2

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (view_a, view_b) in enumerate(dataloader):
            view_a, view_b = view_a.to(device, non_blocking=True), view_b.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                _, z_a = model(view_a)
                _, z_b = model(view_b)
                loss = criterion(z_a, z_b) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        epoch_loss = running_loss / len(dataloader)
        historique_loss.append(epoch_loss)
        print(f"SimCLR Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modèle SimCLR sauvegardé sous : {save_path}")
    return model

def train_segmentation(seg_model, dataloader, device, epochs, save_path):
    print("\n--- DÉBUT DE LA PHASE 2 : FINE-TUNING SEGMENTATION ---")
    
    # Perte adaptée aux masques binaires de ton dataset
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(seg_model.parameters(), lr=1e-4) # Learning rate plus faible pour le fine-tuning
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        seg_model.train()
        running_loss = 0.0
        
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = seg_model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Segmentation Epoch [{epoch+1}/{epochs}] | BCE Loss: {epoch_loss:.4f}")

    torch.save(seg_model.state_dict(), save_path)
    print(f"Modèle de Segmentation sauvegardé sous : {save_path}")

# =====================================================================
# 4. EXÉCUTION PRINCIPALE
# =====================================================================

def main():
    dataset_dir = "./dataset_brats"
    tar_path = "./BraTS2021_Training_Data.tar"
    
    # Décompression si besoin
    if not os.path.exists(dataset_dir):
        print("Décompression...")
        os.makedirs(dataset_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dataset_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Machine utilisée : {device}")

    # Paramètres globaux
    simclr_epochs = 20
    seg_epochs = 20
    batch_size = 16
    simclr_weights_path = "./encodeur_simclr_brats.pth"
    seg_weights_path = "./modele_segmentation_final.pth"

    # --- PHASE 1 : SIMCLR ---
    encoder = SimCLR_Encoder(out_dim=128).to(device)
    
    if os.path.exists(simclr_weights_path):
        print("\nPoids SimCLR trouvés ! Chargement direct de l'encodeur...")
        encoder.load_state_dict(torch.load(simclr_weights_path))
    else:
        dataset_simclr = BraTS_SimCLR_Dataset(dataset_dir)
        loader_simclr = DataLoader(dataset_simclr, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        encoder = train_simclr(encoder, loader_simclr, device, simclr_epochs, simclr_weights_path)

    # --- PHASE 2 : SEGMENTATION ---
    # On intègre l'encodeur pré-entraîné dans notre modèle de segmentation
    seg_model = SegmentationModel(encoder).to(device)
    
    dataset_seg = BraTS2DDataset(dataset_dir)
    loader_seg = DataLoader(dataset_seg, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    train_segmentation(seg_model, loader_seg, device, seg_epochs, seg_weights_path)
    
    print("\nEntraînement complet terminé avec succès !")

if __name__ == '__main__':
    main()

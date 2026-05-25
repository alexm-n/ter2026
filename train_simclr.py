import os
import glob
import random
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ARCHITECTURES

class SimCLR_Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLR_Encoder, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, out_dim)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        if return_features: return features
        h = self.pool(features)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return h, z

class SegmentationModel(nn.Module):
    def __init__(self, encoder):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        features = self.encoder(x, return_features=True)
        out = self.decoder(features)
        return F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

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

# DATASETS 

augmentation_pipeline = T.Compose([
    T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
])

def add_gaussian_noise(tensor, mean=0., std=0.05):
    return tensor + (torch.randn(tensor.size()) * std + mean)

class BraTS_SimCLR_Dataset(Dataset):
    def __init__(self, patient_list):
        self.patient_folders = patient_list
        
    def __len__(self): return len(self.patient_folders)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_folders[idx]
        patient_id = os.path.basename(patient_dir)
        slice_idx = 75
        
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [nib.load(os.path.join(patient_dir, f"{patient_id}_{m}.nii.gz")).get_fdata()[:, :, slice_idx] for m in modalities]
        
        def normalize(img):
            return (img - img.min()) / (img.max() - img.min()) if img.max() > 0 else img
        
        imgs = list(map(normalize, imgs))
        base_image = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)

        view_a = add_gaussian_noise(augmentation_pipeline(base_image))
        view_b = add_gaussian_noise(augmentation_pipeline(base_image))
        return view_a, view_b

class BraTS2DDataset(Dataset):
    def __init__(self, patient_list):
        self.patient_folders = patient_list

    def __len__(self): return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_dir = self.patient_folders[idx]
        patient_id = os.path.basename(patient_dir)
        slice_idx = 75

        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [nib.load(os.path.join(patient_dir, f"{patient_id}_{m}.nii.gz")).get_fdata()[:, :, slice_idx] for m in modalities]
        mask = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")).get_fdata()[:, :, slice_idx]

        def normalize(img): return (img - img.min()) / (img.max() - img.min()) if img.max() > 0 else img

        imgs = list(map(normalize, imgs))
        image = np.stack(imgs, axis=0)

        mask[mask > 0] = 1
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ENTRAINEMENT

def train_simclr(model, train_loader, val_loader, device, epochs, save_path):
    print(f"\nEntrainement SimCLR {epochs} epochs")
    criterion = NTXentLoss(temperature=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    accumulation_steps = 2
    history = []

    for epoch in range(epochs):
        # entraînement
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (view_a, view_b) in enumerate(train_loader):
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

            running_train_loss += loss.item() * accumulation_steps

        # validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for view_a, view_b in val_loader:
                view_a, view_b = view_a.to(device, non_blocking=True), view_b.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    _, z_a = model(view_a)
                    _, z_b = model(view_b)
                    loss = criterion(z_a, z_b)
                    running_val_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        
        history.append({"Phase": "SimCLR", "Epoch": epoch+1, "Train_Loss": epoch_train_loss, "Val_Loss": epoch_val_loss})
        print(f"SimCLR Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    return model, history

# FINE TUNING

def train_segmentation(seg_model, train_loader, val_loader, device, epochs, save_path):
    print(f"\nfine-tuning U-Net {epochs} epochs")
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(seg_model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    
    for epoch in range(epochs):
        # entrainement
        seg_model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = seg_model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()

        # validation
        seg_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = seg_model(images)
                    loss = criterion(outputs, masks)
                    running_val_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        
        history.append({"Phase": "Segmentation", "Epoch": epoch+1, "Train_Loss": epoch_train_loss, "Val_Loss": epoch_val_loss})
        print(f"Seg Epoch [{epoch+1}/{epochs}] | Train BCE: {epoch_train_loss:.4f} | Val BCE: {epoch_val_loss:.4f}")

    torch.save(seg_model.state_dict(), save_path)
    return history

def main():
    # -----------------------------------------------------------------
    POURCENTAGE_A_UTILISER = 80 
    # -----------------------------------------------------------------
    
    BASE_EPOCHS_SIMCLR = 100
    BASE_EPOCHS_SEG = 50
    batch_size = 16

    facteur = 100.0 / POURCENTAGE_A_UTILISER
    simclr_epochs = int(BASE_EPOCHS_SIMCLR * facteur)
    seg_epochs = int(BASE_EPOCHS_SEG * facteur)

    print("==================================================")
    print(f" Debut entrainement sur {POURCENTAGE_A_UTILISER}% du dataset")
    print(f" -> Entrainement SimCLR {simclr_epochs} epochs")
    print(f" -> Fine tuning U-net {seg_epochs} epochs")
    print("==================================================\n")

    train_dir = "./dataset_brats_split/train"
    val_dir = "./dataset_brats_split/val"

    tous_les_patients_train = sorted(glob.glob(os.path.join(train_dir, "BraTS2021_*")))
    random.seed(42)
    random.shuffle(tous_les_patients_train)

    nb_patients_voulus = int((POURCENTAGE_A_UTILISER / 100.0) * len(tous_les_patients_train))
    patients_train_subset = tous_les_patients_train[:max(1, nb_patients_voulus)]
    patients_val = sorted(glob.glob(os.path.join(val_dir, "BraTS2021_*")))

    print(f"Patients pour entrainement : {len(patients_train_subset)}")
    print(f"Patients pour validation  : {len(patients_val)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simclr_weights_path = f"./encodeur_simclr_{POURCENTAGE_A_UTILISER}pct.pth"
    seg_weights_path = f"./modele_segmentation_{POURCENTAGE_A_UTILISER}pct.pth"
    csv_history_path = f"./historique_loss_{POURCENTAGE_A_UTILISER}pct.csv"

    historique_complet = []

    # ENTRAINEMENT
    encoder = SimCLR_Encoder(out_dim=128).to(device)
    
    if os.path.exists(simclr_weights_path):
        print(f"Poids trouvés pour SimCLR {POURCENTAGE_A_UTILISER}%")
        encoder.load_state_dict(torch.load(simclr_weights_path))
    else:
        dataset_simclr_train = BraTS_SimCLR_Dataset(patients_train_subset)
        dataset_simclr_val = BraTS_SimCLR_Dataset(patients_val)
        
        loader_simclr_train = DataLoader(dataset_simclr_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        loader_simclr_val = DataLoader(dataset_simclr_val, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        
        encoder, hist_simclr = train_simclr(encoder, loader_simclr_train, loader_simclr_val, device, simclr_epochs, simclr_weights_path)
        historique_complet.extend(hist_simclr)

    # FINE-TUNING
    seg_model = SegmentationModel(encoder).to(device)
    
    dataset_seg_train = BraTS2DDataset(patients_train_subset)
    dataset_seg_val = BraTS2DDataset(patients_val)
    
    loader_seg_train = DataLoader(dataset_seg_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    loader_seg_val = DataLoader(dataset_seg_val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    hist_seg = train_segmentation(seg_model, loader_seg_train, loader_seg_val, device, seg_epochs, seg_weights_path)
    historique_complet.extend(hist_seg)

    # SAUVEGARDE
    if len(historique_complet) > 0:
        df = pd.DataFrame(historique_complet)
        df.to_csv(csv_history_path, index=False, sep=";")
        print(f"\nHistorique Train vs Val sauvegardé dans : {csv_history_path}")

    print("\nsuccess!")

if __name__ == '__main__':
    main()
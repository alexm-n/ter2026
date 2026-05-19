import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt

# =====================================================================
# 1. REDÉFINITION DES ARCHITECTURES (Obligatoire pour charger les poids)
# =====================================================================

class SimCLR_Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLR_Encoder, self).__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        if return_features: return features
        h = self.pool(features)
        h = torch.flatten(h, 1)
        return h, self.projector(h)

class SegmentationModel(nn.Module):
    def __init__(self, encoder):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
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
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x, return_features=True)
        out = self.decoder(features)
        return F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

# =====================================================================
# 2. DATASET DE TEST
# =====================================================================

class BraTS2DDataset(Dataset):
    def __init__(self, data_dir):
        self.patient_folders = glob.glob(os.path.join(data_dir, "BraTS2021_*"))

    def __len__(self): return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_dir = self.patient_folders[idx]
        patient_id = os.path.basename(patient_dir)
        slice_idx = 75 # On garde la même coupe pour comparer

        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = [nib.load(os.path.join(patient_dir, f"{patient_id}_{m}.nii.gz")).get_fdata()[:, :, slice_idx] for m in modalities]
        mask = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")).get_fdata()[:, :, slice_idx]

        def normalize(img):
            return (img - img.min()) / (img.max() - img.min()) if img.max() > 0 else img

        imgs = list(map(normalize, imgs))
        image = np.stack(imgs, axis=0)

        mask[mask > 0] = 1
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), patient_id

# =====================================================================
# 3. MÉTRIQUE MATHÉMATIQUE (Score de Dice)
# =====================================================================

def compute_dice_score(pred, target, epsilon=1e-6):
    """ Calcule le score de Dice entre le masque prédit et le vrai masque """
    pred = (pred > 0.5).float() # Binarisation
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

# =====================================================================
# 4. FONCTION DE VISUALISATION INTERACTIVE
# =====================================================================

def main():
    dataset_dir = "./dataset_brats"
    weights_path = "./modele_segmentation_final.pth"
    
    if not os.path.exists(weights_path):
        print(f"Erreur : Le fichier de poids '{weights_path}' n'a pas été trouvé.")
        print("Attends que ton premier script termine son entraînement avant de lancer celui-ci.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}")

    # 1. Reconstruction du modèle et chargement des poids sauvegardés
    base_encoder = SimCLR_Encoder(out_dim=128)
    model = SegmentationModel(base_encoder).to(device)
    
    print("Chargement des poids du U-Net...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Mode évaluation

    # 2. Chargement du dataset
    dataset = BraTS2DDataset(dataset_dir)
    # On met shuffle=True pour voir des patients différents à chaque fois qu'on lance le script
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) 

    # 3. Récupération d'un lot d'images
    images, masks, patient_ids = next(iter(dataloader))
    images_dev = images.to(device)
    
    with torch.no_grad():
        outputs = model(images_dev)
        preds_prob = torch.sigmoid(outputs) # Convertit en probabilités [0, 1]

    # 4. Affichage avec Matplotlib
    num_images = images.shape[0]
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 3.5 * num_images))

    print("\n--- ÉVALUATION ET GÉNÉRATION DES IMAGES ---")
    for i in range(num_images):
        img_flair = images[i, 3, :, :].numpy() # Modalité FLAIR
        mask_gt = masks[i, 0, :, :].numpy()
        pred_prob_img = preds_prob[i, 0, :, :].cpu()
        mask_pred = (pred_prob_img > 0.5).float().numpy()

        # Calcul du score de Dice pour ce patient spécifique
        dice = compute_dice_score(pred_prob_img, masks[i, 0, :, :])
        print(f"Patient {patient_ids[i]} | Score de Dice : {dice.item()*100:.2f}%")

        # Colonne 1 : Image FLAIR brute
        axes[i, 0].imshow(img_flair, cmap='gray')
        axes[i, 0].set_title(f"Patient: {patient_ids[i]}\n(FLAIR)")
        axes[i, 0].axis('off')

        # Colonne 2 : Vérité Terrain (Ground Truth en Bleu)
        axes[i, 1].imshow(img_flair, cmap='gray')
        axes[i, 1].imshow(mask_gt, cmap='Blues', alpha=0.4)
        axes[i, 1].set_title("Masque Réel\n(Dessiné par l'expert)")
        axes[i, 1].axis('off')

        # Colonne 3 : Prédiction (Modèle en Rouge) avec son score
        axes[i, 2].imshow(img_flair, cmap='gray')
        axes[i, 2].imshow(mask_pred, cmap='Reds', alpha=0.4)
        axes[i, 2].set_title(f"Prédiction U-Net\nDice Score: {dice.item()*100:.1f}%")
        axes[i, 2].axis('off')

    plt.tight_layout()
    
    # En local, on ouvre une vraie fenêtre interactive sur ton écran !
    print("\nOuverture de la fenêtre graphique... (Ferme la fenêtre pour arrêter le script)")
    plt.show()

if __name__ == '__main__':
    main()
import os
import glob
import shutil
import random

def main():
    source_dir = "./dataset_brats"
    output_dir = "./dataset_brats_split"

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    patients = glob.glob(os.path.join(source_dir, "BraTS2021_*"))
    patients.sort() 
    
    total = len(patients)
    print(f"Nombre total de patients trouvés : {total}")

    if total == 0:
        print("Erreur : Aucun patient trouvé")
        return

    random.seed(42)
    random.shuffle(patients)

    # 70% Train 15% Val 15% Test)
    nb_val = int(0.15 * total)
    nb_test = int(0.15 * total)
  
    
    patients_test = patients[:nb_test]
    patients_val = patients[nb_test : nb_test + nb_val]
    patients_train = patients[nb_test + nb_val :]

    print(f"{len(patients_train)} Train | {len(patients_val)} Val | {len(patients_test)} Test")

    def deplacer_dossiers(liste_patients, dossier_cible, nom_split):
        for patient_path in liste_patients:
            patient_id = os.path.basename(patient_path)
            destination = os.path.join(dossier_cible, patient_id)
            
            shutil.move(patient_path, destination)

    deplacer_dossiers(patients_test, test_dir, "TEST")
    deplacer_dossiers(patients_val, val_dir, "VALIDATION")
    deplacer_dossiers(patients_train, train_dir, "TRAIN")


    print("\nsuccess!")
    print(f"output : {output_dir}")

if __name__ == '__main__':
    main()
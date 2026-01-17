import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from eigenface import train_eigenfaces # On suppose que cette fonction marche bien

matplotlib.use('TkAgg')

def load_image(folder_path):
    images = []
    person = []
    img_shape = None
    
    person_folders = os.listdir(folder_path)
    
    for person_image in person_folders:
        person_path = os.path.join(folder_path, person_image)
        if not os.path.isdir(person_path):
            continue
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, 0) # Lecture en gris
            
            if img is None:
                continue
            if img_shape is None:
                img_shape = img.shape
                
            img_flat = img.flatten()
            images.append(img_flat)
            person.append(person_image)
            
    images = np.array(images, dtype=np.float32)
    person = np.array(person)
    print(f"Dimension de la matrice X : {images.shape}")
    
    return images, person, img_shape

# --- AJOUT 1 : FONCTION DE PROJECTION ---
def get_weights(data, mean_face, eigenfaces):
    """Calcule la signature (poids) des images"""
    data_centered = data - mean_face
    # Projection : (N, pixels) dot (pixels, num_components)
    weights = np.dot(data_centered, eigenfaces)
    return weights

# --- AJOUT 2 : FONCTION DE PRÉDICTION ---
def predict_face(test_image, mean_face, eigenfaces, train_weights, train_labels):
    """Reconnaît une image inconnue"""
    # 1. Centrer et Projeter l'image test
    test_image_centered = test_image - mean_face
    test_weight = np.dot(test_image_centered.reshape(1, -1), eigenfaces)
    
    # 2. Calculer les distances avec toutes les signatures connues
    distances = np.linalg.norm(train_weights - test_weight, axis=1)
    
    # 3. Trouver le plus proche
    min_dist_index = np.argmin(distances)
    min_dist = distances[min_dist_index]
    predicted_label = train_labels[min_dist_index]
    
    return predicted_label, min_dist

# --- MAIN BLOCK MODIFIÉ ---
if __name__ == "__main__":
    chemin_dataset = "./face_database"  
    
    try:
        # 1. Chargement
        images, labels, shape = load_image(chemin_dataset)
        
        # 2. Séparation TRAIN (Apprentissage) / TEST (Validation)
        # On garde les 10 dernières images pour tester si ça marche
        num_test_images = 10
        
        X_train = images[:-num_test_images]
        y_train = labels[:-num_test_images]
        
        X_test = images[-num_test_images:]
        y_test = labels[-num_test_images:]
        
        print(f"Entraînement sur {len(X_train)} images. Test sur {len(X_test)} images.")

        # 3. Entraînement (Calcul des Eigenfaces)
        # Note : On entraîne UNIQUEMENT sur X_train
        mean_face, eigenfaces, X_centered = train_eigenfaces(X_train, n_components=50)

        # 4. Création des signatures (Projection des données connues)
        train_weights = get_weights(X_train, mean_face, eigenfaces)

        # 5. Visualisation (Visage Moyen et Eigenface)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mean_face.reshape(shape), cmap='gray') # Correction: 'shape' ici
        plt.title("Le Visage Moyen")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(eigenfaces[:, 0].reshape(shape), cmap='gray') # Correction: 'shape' ici
        plt.title("Eigenface #1 (La plus importante)")
        plt.axis('off')
        plt.show(block=False) # block=False permet au code de continuer
        plt.pause(2) # Pause de 2 secondes pour voir l'image
        plt.close()

        # 6. Boucle de TEST de reconnaissance
        print("\n--- DÉBUT DU TEST ---")
        correct = 0
        for i in range(len(X_test)):
            label_predit, distance = predict_face(X_test[i], mean_face, eigenfaces, train_weights, y_train)
            vrai_label = y_test[i]
            
            status = "OK" if label_predit == vrai_label else "ERREUR"
            print(f"Image {i+1}: Vrai={vrai_label} | Predit={label_predit} (Dist={distance:.2f}) -> {status}")
            
            if label_predit == vrai_label:
                correct += 1
                
        precision = (correct / len(X_test)) * 100
        print(f"\nPRÉCISION TOTALE : {precision:.2f}%")

    except Exception as e:
        print(f"Erreur critique : {e}")
        import traceback
        traceback.print_exc()
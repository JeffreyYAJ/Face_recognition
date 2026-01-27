import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from eigenface import train_eigenfaces 
matplotlib.use('TkAgg')

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues") # annot=True si peu de classes
    plt.xlabel('Predit')
    plt.ylabel('Vrai')
    plt.title('Matrice de Confusion')
    plt.show()


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
            img = cv2.imread(image_path, 0) 
            
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

# --- Projection fonction ---
def get_weights(data, mean_face, eigenfaces):
    """Calcule la signature (poids) des images"""
    data_centered = data - mean_face
    
    weights = np.dot(data_centered, eigenfaces)
    return weights

# ---  Prediction fonction ---
def predict_face(test_image, mean_face, eigenfaces, train_weights, train_labels):
    test_image_centered = test_image - mean_face
    test_weight = np.dot(test_image_centered.reshape(1, -1), eigenfaces)
    
    distances = np.linalg.norm(train_weights - test_weight, axis=1)
    
    min_dist_index = np.argmin(distances)
    min_dist = distances[min_dist_index]
    predicted_label = train_labels[min_dist_index]
    
    return predicted_label, min_dist

def analyze_performance(X_train, y_train, X_test, y_test):
    components_list = [5, 10, 20, 30, 50, 100]
    accuracies = []
    
    for n in components_list:
        print(f"Test avec {n} composantes...")
        mean, eigenfaces, _ = train_eigenfaces(X_train, n)
        train_weights = get_weights(X_train, mean, eigenfaces)
        
        correct = 0
        for i in range(len(X_test)):
            pred, _ = predict_face(X_test[i], mean, eigenfaces, train_weights, y_train)
            if pred == y_test[i]:
                correct += 1
        
        acc = correct / len(X_test)
        accuracies.append(acc)
        
    plt.plot(components_list, accuracies, marker='o')
    plt.xlabel("Nombre d'Eigenfaces")
    plt.ylabel("Précision")
    plt.title("Impact de la réduction de dimension sur la précision")
    plt.grid()
    plt.show()
    
def reconstruct_image(image_originale, mean_face, eigenfaces):
    image_centered = image_originale.flatten() - mean_face
    weights = np.dot(image_centered, eigenfaces)
    
    reconstruction = np.dot(weights, eigenfaces.T) + mean_face
    
    return reconstruction

if __name__ == "__main__":
    chemin_dataset = "./face_database"  
    
    try: 
        images, labels, shape = load_image(chemin_dataset)
        
        np.random.seed(42) 
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]
        
        # 3. Séparation Train / Test
        num_test_image = 50
        X_train = images[:-num_test_image]
        y_train = labels[:-num_test_image]
        X_test = images[-num_test_image:]
        y_test = labels[-num_test_image:]
        
        # 4. Entraînement (Calcul des Eigenfaces)
        print("--- Entraînement en cours ---")
        mean_face, eigenfaces, X_centered = train_eigenfaces(X_train, n_components=50)
        train_weights = get_weights(X_train, mean_face, eigenfaces)

        # 5. Visualisation (Visage Moyen et Eigenface)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mean_face.reshape(shape), cmap='gray')
        plt.title("Le Visage Moyen")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(eigenfaces[:, 0].reshape(shape), cmap='gray') 
        plt.title("Eigenface #1 (Ghost Face)")
        plt.axis('off')
        plt.show(block=False) 
        plt.pause(2) # On attend 2 secondes puis on continue
        plt.close()
        
        # 6. Test de Reconnaissance Standard
        print("\n--- DÉBUT DU TEST DE RECONNAISSANCE ---")
        correct = 0
        for i in range(len(X_test)):
            label_predit, distance = predict_face(X_test[i], mean_face, eigenfaces, train_weights, y_train)
            vrai_label = y_test[i]
            
            # Affichage allégé pour ne pas spammer la console
            if i < 5: # On affiche juste les 5 premiers détails
                status = "OK" if label_predit == vrai_label else "ERREUR"
                print(f"Image {i+1}: Vrai={vrai_label} | Predit={label_predit} (Dist={distance:.2f}) -> {status}")
            
            if label_predit == vrai_label:
                correct += 1
                
        precision = (correct / len(X_test)) * 100
        print(f"\n>>> PRÉCISION FINALE (50 composantes) : {precision:.2f}%")

        # 7. PARTIE RECHERCHE : Courbe de performance
        # C'est ici qu'on utilise la fonction que vous avez ajoutée !
        print("\n--- ANALYSE DE PERFORMANCE (Courbe) ---")
        print("Calcul en cours pour 5, 10, 20... composantes. Patientez.")
        analyze_performance(X_train, y_train, X_test, y_test)
        
        # 8. PARTIE MATHÉMATIQUE : Reconstruction
        # On montre comment le visage est reconstruit
        print("\n--- DÉMONSTRATION DE RECONSTRUCTION ---")
        image_test = X_test[0] # On prend la première image de test
        reconstruction = reconstruct_image(image_test, mean_face, eigenfaces)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_test.reshape(shape), cmap='gray')
        plt.title(f"Originale ({y_test[0]})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction.reshape(shape), cmap='gray')
        plt.title("Reconstruite avec 50 Eigenfaces")
        plt.axis('off')
        plt.show() 
    except Exception as e:
        print(f"Erreur critique : {e}")
        import traceback
        traceback.print_exc()
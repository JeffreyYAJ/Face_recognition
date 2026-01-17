import numpy as np

def train_eigenfaces(X, n_components=50):
    
    print("1. Calcul du visage moyen...")
    mean_face = np.mean(X, axis=0) 
    
    print("2. Centrage des données (Soustraction de la moyenne)...")
    A = X - mean_face 
    
    print("3. Calcul de la Matrice de Covariance (L'astuce !)...")
    C_small = np.dot(A, A.T)
    
    print("4. Calcul des Valeurs Propres et Vecteurs Propres...")
    eigenvalues, eigenvectors_small = np.linalg.eigh(C_small)
    
    sorted_index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors_small = eigenvectors_small[:, sorted_index]
    
    eigenvalues = eigenvalues[:n_components]
    eigenvectors_small = eigenvectors_small[:, :n_components]
    
    print(f"5. Récupération des 'Vrais' Vecteurs Propres (Eigenfaces)...")
    eigenfaces = np.dot(A.T, eigenvectors_small)
    
    print("6. Normalisation des Eigenfaces...")
    for i in range(eigenfaces.shape[1]):
        eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
        
    print(f"Entraînement terminé ! Nous avons {eigenfaces.shape[1]} Eigenfaces.")
    
    return mean_face, eigenfaces, A

# TEST
# if __name__ == "__main__":
    
#     mean_face, eigenfaces, X_centered = train_eigenfaces(X, n_components=50)

#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(mean_face.reshape(img_shape), cmap='gray')
#     plt.title("Le Visage Moyen (Mean Face)")
#     plt.axis('off')

#     # Affichons la première Eigenface (la caractéristique la plus dominante)
#     plt.subplot(1, 2, 2)
#     plt.imshow(eigenfaces[:, 0].reshape(img_shape), cmap='gray')
#     plt.title("Eigenface #1 (Ghost Face)")
#     plt.axis('off')
    
#     plt.show()
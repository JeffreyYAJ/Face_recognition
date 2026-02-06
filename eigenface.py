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


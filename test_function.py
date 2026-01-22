def analyze_performance(X_train, y_train, X_test, y_test):
    components_list = [5, 10, 20, 30, 50, 100]
    accuracies = []
    
    for n in components_list:
        print(f"Test avec {n} composantes...")
        # Entraînement
        mean, eigenfaces, _ = train_eigenfaces(X_train, n)
        train_weights = get_weights(X_train, mean, eigenfaces)
        
        # Test
        correct = 0
        for i in range(len(X_test)):
            pred, _ = predict_face(X_test[i], mean, eigenfaces, train_weights, y_train)
            if pred == y_test[i]:
                correct += 1
        
        acc = correct / len(X_test)
        accuracies.append(acc)
        
    # Plot
    plt.plot(components_list, accuracies, marker='o')
    plt.xlabel("Nombre d'Eigenfaces")
    plt.ylabel("Précision")
    plt.title("Impact de la réduction de dimension sur la précision")
    plt.grid()
    plt.show()
    
def reconstruct_image(image_originale, mean_face, eigenfaces):
    # 1. Obtenir la signature (poids)
    image_centered = image_originale.flatten() - mean_face
    weights = np.dot(image_centered, eigenfaces)
    
    # 2. Reconstruire (Inverse) : Poids * Eigenfaces + Moyenne
    # (1, 50) x (50, 10304) -> (1, 10304)
    reconstruction = np.dot(weights, eigenfaces.T) + mean_face
    
    return reconstruction
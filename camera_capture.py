import os
import cv2
import numpy as np
import time
from eigenface import train_eigenfaces

def load_image(folder_path):
    """Charge les images de la base de données"""
    images = []
    person = []
    img_shape = None
    
    person_folders = sorted([d for d in os.listdir(folder_path) 
                            if os.path.isdir(os.path.join(folder_path, d))])
    
    for person_image in person_folders:
        person_path = os.path.join(folder_path, person_image)
        
        for image_name in sorted(os.listdir(person_path)):
            image_path = os.path.join(person_path, image_name)
            if not image_name.lower().endswith(('.pgm', '.jpg', '.png')):
                continue
                
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
    
    return images, person, img_shape

def get_weights(data, mean_face, eigenfaces):
    """Calcule la signature (poids) des images"""
    data_centered = data - mean_face
    weights = np.dot(data_centered, eigenfaces)
    return weights

def predict_face(test_image, mean_face, eigenfaces, train_weights, train_labels, threshold=5000):
    """Prédit le visage d'une image de test avec un seuil"""
    test_image_flat = test_image.flatten().astype(np.float32)
    test_image_centered = test_image_flat - mean_face
    test_weight = np.dot(test_image_centered.reshape(1, -1), eigenfaces)
    
    distances = np.linalg.norm(train_weights - test_weight, axis=1)
    
    min_dist_index = np.argmin(distances)
    min_dist = distances[min_dist_index]
    predicted_label = train_labels[min_dist_index]
    
    # Si la distance est trop grande, c'est un visage inconnu
    if min_dist > threshold:
        return "INCONNU", min_dist
    
    return predicted_label, min_dist

def run_face_recognition_camera(dataset_path="./face_database", n_components=50, threshold=5000):
    """Lance la reconnaissance faciale en temps réel sur la caméra"""
    
    print("🔄 Chargement de la base de données...")
    try:
        images, labels, img_shape = load_image(dataset_path)
        print(f"✓ {len(images)} images chargées")
        print(f"✓ Dimensions des visages: {img_shape}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    print(f"🔄 Entraînement des Eigenfaces (n_components={n_components})...")
    try:
        mean_face, eigenfaces, _ = train_eigenfaces(images, n_components=n_components)
        train_weights = get_weights(images, mean_face, eigenfaces)
        print("✓ Modèle entraîné avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return
    
    print("📷 Initialisation de la caméra...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra!")
        return
    
    # Charger le classificateur en cascade pour la détection de visages
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print("✓ Caméra prête!")
    print("Appuyez sur 'q' pour quitter")
    print("-" * 50)
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    recognized_count = 0
    unknown_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Erreur lors de la lecture du flux vidéo")
            break
        
        # Calculer les FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0.5:  # Mettre à jour tous les 0.5 secondes
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Afficher les informations
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(frame, f"Visages detectes: {len(faces)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_recognized_this = 0
        frame_unknown_this = 0
        
        for (x, y, w, h) in faces:
            # 1. Découper le visage
            roi_gray = gray[y:y+h, x:x+w]
            
            # 2. Redimensionner à la taille du dataset
            roi_resized = cv2.resize(roi_gray, (img_shape[1], img_shape[0]))
            
            # 3. Prédire
            try:
                label, dist = predict_face(roi_resized, mean_face, eigenfaces, 
                                         train_weights, labels, threshold=threshold)
                
                # 4. Déterminer la couleur en fonction du résultat
                if label == "INCONNU":
                    color = (0, 0, 255)  # Rouge pour inconnu
                    display_text = f"INCONNU (dist: {dist:.1f})"
                    frame_unknown_this += 1
                    unknown_count += 1
                else:
                    color = (0, 255, 0)  # Vert pour reconnu
                    display_text = f"{label} (dist: {dist:.1f})"
                    frame_recognized_this += 1
                    recognized_count += 1
                
                # 5. Dessiner le rectangle et le label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, display_text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "ERREUR", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Afficher les statistiques
        if frame_recognized_this > 0:
            cv2.putText(frame, f"Reconnus: {frame_recognized_this}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if frame_unknown_this > 0:
            cv2.putText(frame, f"Inconnus: {frame_unknown_this}", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Afficher les instructions
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Reconnaissance Faciale en Temps Reel', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✓ Fermeture de l'application...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Afficher les statistiques finales
    if recognized_count + unknown_count > 0:
        total = recognized_count + unknown_count
        print(f"\n📊 Statistiques:")
        print(f"   Visages reconnus: {recognized_count}")
        print(f"   Visages inconnus: {unknown_count}")
        print(f"   Total: {total}")
        recognition_rate = (recognized_count / total) * 100
        print(f"   Taux de reconnaissance: {recognition_rate:.1f}%")
    
    print("✓ Terminé!")

if __name__ == "__main__":
    import os
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=50,
        threshold=5000  # Ajustez ce seuil selon vos besoins
    )
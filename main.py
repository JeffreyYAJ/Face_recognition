import os
import cv2
import numpy as np

import matplotlib; 

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
            
            img = cv2.imread(image_path,0)
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

if __name__ == "__main__":
    chemin_dataset = "./face_database"  
    
    try:
        images, labels, shape = load_image(chemin_dataset)
        print(f"Taille d'une image originale : {shape}")
        
        
        import matplotlib.pyplot as plt
        plt.imshow(images[0].reshape(shape), cmap='gray')
        plt.title(f"Personne : {labels[0]}")
        plt.show()
        
    except Exception as e:
        print(f"Erreur : {e}")
  
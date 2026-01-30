#!/usr/bin/env python3
"""
Script de test: Vérifier que tous les modules se chargent correctement
"""

import sys
import os

print("=" * 60)
print("  TEST DES MODULES")
print("=" * 60)

# Test des imports
tests = []

print("\n1️⃣  Vérification des imports...")
try:
    import cv2
    print("   ✓ OpenCV (cv2)")
    tests.append(True)
except ImportError as e:
    print(f"   ❌ OpenCV: {e}")
    tests.append(False)

try:
    import numpy
    print("   ✓ NumPy")
    tests.append(True)
except ImportError as e:
    print(f"   ❌ NumPy: {e}")
    tests.append(False)

try:
    import matplotlib
    print("   ✓ Matplotlib")
    tests.append(True)
except ImportError as e:
    print(f"   ❌ Matplotlib: {e}")
    tests.append(False)

try:
    import sklearn
    print("   ✓ Scikit-learn")
    tests.append(True)
except ImportError as e:
    print(f"   ❌ Scikit-learn: {e}")
    tests.append(False)

try:
    import seaborn
    print("   ✓ Seaborn")
    tests.append(True)
except ImportError as e:
    print(f"   ❌ Seaborn: {e}")
    tests.append(False)

print("\n2️⃣  Vérification des modules locaux...")
try:
    from eigenface import train_eigenfaces
    print("   ✓ eigenface.py")
    tests.append(True)
except Exception as e:
    print(f"   ❌ eigenface.py: {e}")
    tests.append(False)

try:
    from camera_capture import run_face_recognition_camera
    print("   ✓ camera_capture.py")
    tests.append(True)
except Exception as e:
    print(f"   ❌ camera_capture.py: {e}")
    tests.append(False)

try:
    import main
    print("   ✓ main.py")
    tests.append(True)
except Exception as e:
    print(f"   ❌ main.py: {e}")
    tests.append(False)

try:
    import config
    print("   ✓ config.py")
    tests.append(True)
except Exception as e:
    print(f"   ❌ config.py: {e}")
    tests.append(False)

print("\n3️⃣  Vérification de la base de données...")
dataset_path = "./face_database"
if os.path.isdir(dataset_path):
    folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"   ✓ Base de données trouvée ({len(folders)} dossiers)")
    
    # Compter les images
    total_images = 0
    for folder in folders:
        images = [f for f in os.listdir(os.path.join(dataset_path, folder)) 
                 if f.lower().endswith(('.pgm', '.jpg', '.png'))]
        total_images += len(images)
    
    print(f"   ✓ {total_images} images trouvées")
    tests.append(True)
else:
    print(f"   ❌ Base de données non trouvée ({dataset_path})")
    tests.append(False)

print("\n4️⃣  Vérification de la caméra...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✓ Caméra accessible")
        cap.release()
        tests.append(True)
    else:
        print("   ⚠️  Caméra non accessible (pourrait être normal en headless)")
        tests.append(None)
except Exception as e:
    print(f"   ❌ Erreur caméra: {e}")
    tests.append(False)

# Résumé
print("\n" + "=" * 60)
passed = sum(1 for t in tests if t is True)
failed = sum(1 for t in tests if t is False)
warnings = sum(1 for t in tests if t is None)

print(f"Résultats: {passed} ✓ | {failed} ❌ | {warnings} ⚠️")

if failed == 0:
    print("\n✅ Tous les tests sont passés!")
    print("\nVous pouvez maintenant lancer:")
    print("  python main.py          # Menu interactif")
    print("  python run_camera.py    # Reconnaissance caméra directe")
    sys.exit(0)
else:
    print(f"\n❌ {failed} test(s) échoué(s)")
    print("\nVeuillez installer les dépendances:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

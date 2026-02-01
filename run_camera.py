#!/usr/bin/env python3
"""
Script d'exemple: Reconnaissance Faciale en Temps Réel
Lance directement la reconnaissance faciale sans passer par le menu
"""

from camera_capture import run_face_recognition_camera

if __name__ == "__main__":
    print("=" * 60)
    print("  RECONNAISSANCE FACIALE EN TEMPS RÉEL")
    print("=" * 60)
    print("\nDémarrage de la caméra...")
    print("Appuyez sur 'q' pour quitter\n")
    
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=50,
        threshold=5000
    )
    
    print("\n✓ Application fermée")

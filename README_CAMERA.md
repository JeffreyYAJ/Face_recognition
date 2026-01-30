# Reconnaissance Faciale en Temps Réel

## Description
Ce projet implémente un système de reconnaissance faciale utilisant la méthode des **Eigenfaces** (analyse en composantes principales). Le système peut fonctionner en deux modes:

1. **Mode Analyse**: Test complet avec graphiques et performance
2. **Mode Caméra**: Reconnaissance faciale en temps réel via la webcam

## Installation

### Dépendances
```bash
pip install -r requirements.txt
```

### Dépendances requises:
- numpy
- opencv-python
- matplotlib
- scikit-learn
- seaborn

## Utilisation

### Lancer le programme principal
```bash
python main.py
```

Vous verrez un menu interactif:
```
============================================================
  SYSTÈME DE RECONNAISSANCE FACIALE PAR EIGENFACES
============================================================

1. Exécuter l'analyse complète (Test + Graphiques)
2. Lancer la reconnaissance faciale en temps réel (Caméra)
3. Quitter

============================================================
```

### Option 1: Analyse Complète
Choisissez **1** pour:
- Charger la base de données de visages
- Entraîner le modèle Eigenfaces
- Visualiser le visage moyen et les Eigenfaces
- Afficher les résultats de reconnaissance sur les images de test
- Générer un graphique de performance
- Montrer la reconstruction d'images

### Option 2: Reconnaissance Faciale en Temps Réel
Choisissez **2** pour:
- Démarrer la caméra
- Détecter les visages en temps réel
- Reconnaître les visages connus
- Afficher les noms des personnes reconnues avec un score de confiance

**Contrôles:**
- **q**: Quitter l'application
- Les visages reconnus s'affichent en **vert**
- Les visages inconnus s'affichent en **rouge**

## Architecture du Projet

### Fichiers principaux:

1. **eigenface.py**
   - Implémentation de l'algorithme Eigenfaces
   - Calcul des valeurs propres et vecteurs propres
   - Normalisation des Eigenfaces

2. **camera_capture.py**
   - Gestion de la caméra
   - Détection des visages avec Haar Cascades
   - Reconnaissance faciale en temps réel
   - Fonction `run_face_recognition_camera()` principale

3. **main.py**
   - Menu principal interactif
   - Gestion du flux d'entraînement et test
   - Visualisation des résultats

## Paramètres Configurables

Dans `camera_capture.py`, fonction `run_face_recognition_camera()`:

```python
run_face_recognition_camera(
    dataset_path="./face_database",    # Chemin vers la base de données
    n_components=50,                   # Nombre d'Eigenfaces (50 par défaut)
    threshold=5000                     # Seuil de distance pour reconnaître un visage
)
```

### Explication des paramètres:
- **n_components**: Plus la valeur est haute, plus le modèle capture de détails (mais risque de surapprentissage)
- **threshold**: Si la distance est > au seuil, le visage est considéré comme inconnu
  - Augmentez pour accepter plus de visages
  - Diminuez pour être plus strict

## Base de Données

La base de données doit avoir la structure suivante:
```
face_database/
    s1/
        1.pgm
        2.pgm
        ...
    s2/
        1.pgm
        2.pgm
        ...
```

Chaque dossier `sN` contient les images d'une personne.

## Fonctionnalités Implémentées

✓ Chargement des images de la base de données
✓ Entraînement du modèle Eigenfaces
✓ Détection de visages avec Haar Cascades
✓ Reconnaissance faciale sur images statiques
✓ **Reconnaissance faciale en temps réel sur caméra**
✓ Calcul des distances euclidienne
✓ Seuil de confiance pour les visages inconnus
✓ Visualisation des Eigenfaces
✓ Reconstruction d'images
✓ Analyse de performance
✓ Menu interactif

## Troubleshooting

### La caméra ne fonctionne pas
```bash
# Vérifiez que votre caméra est accessible
v4l2-ctl --list-devices
```

### ImportError: No module named 'opencv'
```bash
# Utilisez opencv-python, pas opencv
pip install opencv-python
```

### Mauvaise reconnaissance
- Ajustez le paramètre `threshold` (augmentez pour être moins strict)
- Augmentez `n_components` pour plus de précision
- Ajoutez plus d'images à la base de données

## Performance

Sur un dataset de 40 personnes (9 images chacune = 360 images):
- Entraînement: ~2-3 secondes
- Reconnaissance par image: ~10-50ms
- Reconnaissance en temps réel: 25-30 FPS

## Améliorations Futures

- [ ] Support de multiples détecteurs de visages
- [ ] Utilisation de CNN (Convolutional Neural Networks)
- [ ] Sauvegarde du modèle entraîné
- [ ] Calibration automatique du threshold
- [ ] Support de plusieurs caméras
- [ ] Interface GUI

## Auteur

Développé pour le projet de reconnaissance faciale.

## Licence

MIT

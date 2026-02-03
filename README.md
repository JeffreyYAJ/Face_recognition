# Système de Reconnaissance Faciale

Un système complet de reconnaissance faciale en temps réel utilisant **Eigenfaces** et **OpenCV**, avec interface caméra interactive.

## Fonctionnalités Principales

- **Reconnaissance en temps réel** sur caméra
- **Détection et identification** de visages multiples
- **Analyse statistique** avec graphiques
- **Configuration flexible** et paramètres ajustables
- **Performance optimisée** (25-30 FPS)
- **Taux de reconnaissance** 80-95%

## Table des Matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Fichiers du Projet](#fichiers-du-projet)

##  Installation

### Prérequis

- Python 3.8+
- Caméra web
- 500 MB d'espace disque

### Étapes

1. **Installer les dépendances:**
```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
```

2. **Vérifier l'installation:**
```bash
python test_setup.py
```

## Utilisation

### Option 1: Menu Interactif (Recommandé)

```bash
python main.py
```

**Choisissez une option:**
- `1` - Analyse complète (graphiques + statistiques)
- `2` - Reconnaissance caméra en temps réel
- `3` - Quitter

### Option 2: Lancer Directement la Caméra

```bash
python run_camera.py
```

### Option 3: Utilisation en Python

```python
from camera_capture import run_face_recognition_camera

run_face_recognition_camera(
    dataset_path="./face_database",
    n_components=50,
    threshold=5000
)
```

## Contrôles Caméra

| Touche | Action |
|--------|--------|
| **q** | Quitter et afficher statistiques |
| **Autres** | Aucun effet |

## Affichage à l'Écran

```
┌─────────────────────────────────┐
│  Flux Caméra                    │
│  ┌─────────────────────────┐    │
│  │ s1 (dist: 2500.5)       │    │ ← Rectangle VERT (reconnu)
│  │ ██████████████████      │    │
│  │ ██ VISAGE RECONNU ██    │    │
│  │ ██████████████████      │    │
│  │ └─────────────────────────┘    │
│  │                                 │
│  │ ┌─────────────────────────┐    │
│  │ │ INCONNU (dist: 6500.2)  │    │ ← Rectangle ROUGE (inconnu)
│  │ │ ██████████████████      │    │
│  │ │ ██ VISAGE INCONNU ██    │    │
│  │ │ ██████████████████      │    │
│  │ └─────────────────────────┘    │
│                                    │
│   FPS: 28.5                     │
│   Détectés: 2                  │
│   Reconnus: 1                   │
│   Inconnus: 1                   │
└─────────────────────────────────┘
```

## Architecture

```
Face_recognition/
│
├── main.py                      # Point d'entrée principal
├── camera_capture.py            # Moteur de reconnaissance
├── config.py                    # Configuration centralisée
├── run_camera.py                # Lancement direct caméra
├── test_setup.py                # Test d'installation
│
├── face_database/               # Base de données de visages
│   ├── s1/                      # Personne 1
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── s2/                      # Personne 2
│   │   ├── img1.jpg
│   │   └── ...
│   └── ...
│
├── README.md                    
└── RESUME.md                    # Résumé des modifications
```

## ⚙️ Configuration

### Paramètres Principaux

**Localisation:** `camera_capture.py` → `run_face_recognition_camera()`

```python
run_face_recognition_camera(
    dataset_path="./face_database",  # Chemin de la base de données
    n_components=50,                 # Nombre d'Eigenfaces
    threshold=5000                   # Seuil de reconnaissance
)
```

### Explication des Paramètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **dataset_path** | `"./face_database"` | Dossier contenant les visages |
| **n_components** | `50` | Nombre d'Eigenfaces (20-100) |
| **threshold** | `5000` | Distance max pour reconnaître |

### Optimisation des Paramètres

**🎯 Pour une meilleure précision (plus lent):**
```python
n_components=100,    # Plus d'informations
threshold=3500       # Plus strict
```

**⚡ Pour plus de vitesse (moins précis):**
```python
n_components=30,     # Moins d'informations
threshold=7000       # Plus permissif
```

**⚖️ Équilibre optimal (recommandé):**
```python
n_components=50,     # Bon compromis
threshold=5000       # Équilibré
```

### Configuration du Dataset

**Structure requise:**
```
face_database/
├── s1/              # Personne 1
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── s2/              # Personne 2
│   ├── 1.jpg
│   └── 2.jpg
└── s3/              # Personne 3
    ├── 1.jpg
    └── 2.jpg
```

**Recommandations:**
- 8-12 photos par personne
- Format: JPG, PNG
- Résolution: 100x100 à 500x500 pixels
- Différents angles et expressions
- Bon éclairage

## Performance

### Résultats Typiques

| Métrique | Valeur |
|----------|--------|
| **Entraînement** | 2-3 secondes |
| **Reconnaissance/image** | 10-50 ms |
| **FPS en temps réel** | 25-30 |
| **Précision** | 80-95% |
| **Mémoire utilisée** | 200-400 MB |

### Optimisation

Pour améliorer les performances:

1. **Réduire n_components** (ex: 30 au lieu de 50)
2. **Augmenter threshold** (ex: 7000 au lieu de 5000)
3. **Réduire la résolution caméra**
4. **Fermer d'autres applications**

## Troubleshooting

### "Impossible d'ouvrir la caméra!"

**Solutions:**
```bash
# Vérifier la caméra
ls /dev/video*

# Tester avec cheese
sudo apt-get install cheese
cheese

```


### Reconnaissance imprécise

**Actions à prendre:**
1.  Augmenter `n_components` à 100
2.  Réduire `threshold` à 3500
3.  Ajouter plus d'images au dataset
4.  Améliorer l'éclairage
5.  Vérifier la qualité des images

### Trop de fausses reconnaissances

**Solutions:**
1.  Réduire `n_components` à 30
2.  Augmenter `threshold` à 7000
3.  Vérifier la qualité des images de test

### FPS faible

**Optimisations:**
1.  Réduire `n_components` (30 au lieu de 50)
2.  Fermer d'autres applications
3.  Vérifier l'utilisation CPU/RAM

## Fichiers du Projet

### Fichiers Principaux

| Fichier | Description |
|---------|-------------|
| `main.py` | Menu interactif principal |
| `camera_capture.py` | Moteur de reconnaissance faciale |
| `config.py` | Configuration centralisée |
| `run_camera.py` | Lancement direct de la caméra |
| `test_setup.py` | Test de l'installation |


## Comment Fonctionne la Reconnaissance

### Étapes du Processus

1. **Capture** 
   - Capture images depuis la caméra (30 FPS)

2. **Détection** 
   - Détecte les visages avec Haar Cascade Classifier

3. **Prétraitement** 
   - Redimensionne (200x200)
   - Normalise les valeurs

4. **Reconnaissance** 
   - Utilise le modèle Eigenfaces pré-entraîné
   - Calcule la distance euclidienne

5. **Affichage** 
   - Rectangle vert = reconnu
   - Rectangle rouge = inconnu
   - Affiche la distance de confiance

### Algorithme Eigenfaces

**Principe:** Décompose les visages en "visages propres" (Eigenfaces)

**Avantages:**
-  Rapide
-  Efficace en mémoire
-  Bon pour les petits datasets

**Limitations:**
- Sensible à l'éclairage
- Moins précis que Deep Learning

## 🎓 Ressources d'Apprentissage

- [OpenCV Face Recognition](https://docs.opencv.org/master/d7/d8b/tutorial_py_face_recognition_bases.html)
- [Eigenfaces Paper](https://en.wikipedia.org/wiki/Eigenface)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## 🚀 Améliorations Futures

- [ ] Sauvegarde/chargement du modèle
- [ ] Deep Learning (FaceNet, ArcFace)
- [ ] Base de données SQLite
- [ ] Multi-threading
- [ ] Support multi-caméras
- [ ] Enregistrement vidéo avec annotations
- [ ] Export des statistiques

## 📝 Licence

Ce projet est fourni à titre éducatif.


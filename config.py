DATABASE_PATH = "./face_database"

N_COMPONENTS = 50  # Nombre d'Eigenfaces à utiliser
                   # Valeurs recommandées: 20-100
                   # Plus haut = plus de détails mais risque de surapprentissage

FACE_DETECTION_SCALE = 1.3  # Facteur d'échelle pour la détection Haar Cascade
FACE_DETECTION_MIN_NEIGHBORS = 5  # Nombre minimum de voisins pour valider une détection
DISTANCE_THRESHOLD = 5000  # Seuil de distance euclidienne
                           # Si distance > seuil, le visage est inconnu
                           

CAMERA_ID = 0  # ID de la caméra (0 = webcam par défaut)
FPS_TARGET = 30  # Images par seconde cibles

RANDOM_SEED = 42  # Pour la reproductibilité
TEST_SET_SIZE = 50  # Nombre d'images de test

SHOW_DISTANCE_SCORE = True  # Afficher le score de distance
SHOW_FPS = True  # Afficher les FPS en temps réel
BOX_COLOR_RECOGNIZED = (0, 255, 0)  
BOX_COLOR_UNKNOWN = (0, 0, 255)  # Couleur pour visages inconnus (BGR)
BOX_THICKNESS = 2  # Épaisseur du rectangle

# Logging
VERBOSE = True  

# Genere par IA
from camera_capture import run_face_recognition_camera

# ============================================================
# Exemple 1: Utilisation Simple (Paramètres par défaut)
# ============================================================
def exemple_simple():
    """Lancer la reconnaissance avec les paramètres par défaut"""
    print("Exemple 1: Configuration simple")
    run_face_recognition_camera()

# ============================================================
# Exemple 2: Mode Strict (Moins de fausses reconnaissances)
# ============================================================
def exemple_strict():
    """Configuration stricte - reconnaît uniquement les visages très proches"""
    print("Exemple 2: Mode strict")
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=50,
        threshold=3000  # Seuil bas = strict
    )

# ============================================================
# Exemple 3: Mode Permissif (Plus de reconnaissances)
# ============================================================
def exemple_permissif():
    """Configuration permissive - accepte plus de variation"""
    print("Exemple 3: Mode permissif")
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=50,
        threshold=7000  # Seuil haut = permissif
    )

# ============================================================
# Exemple 4: Mode Précis (Plus de composantes)
# ============================================================
def exemple_precis():
    """Configuration précise - utilise plus d'Eigenfaces"""
    print("Exemple 4: Mode précis (plus lent mais plus précis)")
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=100,  # Plus de détails
        threshold=5000
    )

# ============================================================
# Exemple 5: Mode Rapide (Moins de composantes)
# ============================================================
def exemple_rapide():
    """Configuration rapide - utilise moins d'Eigenfaces"""
    print("Exemple 5: Mode rapide (moins précis mais plus rapide)")
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=20,  # Moins de détails
        threshold=5000
    )

# ============================================================
# Exemple 6: Configuration Personnalisée
# ============================================================
def exemple_personnalise():
    """Configuration personnalisée"""
    print("Exemple 6: Configuration personnalisée")
    
    # À vous de modifier selon vos besoins!
    run_face_recognition_camera(
        dataset_path="./face_database",
        n_components=75,      # Valeur intermédiaire
        threshold=4500        # Seuil intermédiaire
    )

# ============================================================
# Menu Interactif
# ============================================================
def menu():
    """Menu pour choisir l'exemple"""
    print("\n" + "="*60)
    print("  EXEMPLES AVANCÉS - RECONNAISSANCE FACIALE")
    print("="*60)
    print("\nChoisissez une configuration:")
    print("1. Simple (par défaut)")
    print("2. Strict (moins de fausses reconnaissances)")
    print("3. Permissif (plus de reconnaissances)")
    print("4. Précis (100 Eigenfaces)")
    print("5. Rapide (20 Eigenfaces)")
    print("6. Personnalisé")
    print("0. Quitter")
    print("="*60)
    
    choix = input("\nChoisissez (0-6): ").strip()
    
    if choix == "1":
        exemple_simple()
    elif choix == "2":
        exemple_strict()
    elif choix == "3":
        exemple_permissif()
    elif choix == "4":
        exemple_precis()
    elif choix == "5":
        exemple_rapide()
    elif choix == "6":
        exemple_personnalise()
    elif choix == "0":
        print("Au revoir!")
        return False
    else:
        print("Option invalide")
        return True
    
    return True

# ============================================================
# Tableaux Comparatifs
# ============================================================
def afficher_comparaison():
    """Affiche un tableau comparatif des configurations"""
    print("\n" + "="*80)
    print("  COMPARAISON DES CONFIGURATIONS")
    print("="*80)
    
    configs = [
        {
            "nom": "Simple",
            "n_components": 50,
            "threshold": 5000,
            "vitesse": "Normal",
            "precision": "Bonne",
            "faux_positifs": "Moyen"
        },
        {
            "nom": "Strict",
            "n_components": 50,
            "threshold": 3000,
            "vitesse": "Normal",
            "precision": "Très bonne",
            "faux_positifs": "Très bas"
        },
        {
            "nom": "Permissif",
            "n_components": 50,
            "threshold": 7000,
            "vitesse": "Normal",
            "precision": "Moins bonne",
            "faux_positifs": "Élevé"
        },
        {
            "nom": "Précis",
            "n_components": 100,
            "threshold": 5000,
            "vitesse": "Lent",
            "precision": "Très bonne",
            "faux_positifs": "Bas"
        },
        {
            "nom": "Rapide",
            "n_components": 20,
            "threshold": 5000,
            "vitesse": "Rapide",
            "precision": "Acceptable",
            "faux_positifs": "Élevé"
        }
    ]
    
    # Affichage formaté
    print(f"{'Config':<12} | {'N_Comp':<7} | {'Threshold':<10} | "
          f"{'Vitesse':<8} | {'Précision':<12} | {'Faux +':<10}")
    print("-" * 80)
    
    for config in configs:
        print(f"{config['nom']:<12} | {config['n_components']:<7} | "
              f"{config['threshold']:<10} | {config['vitesse']:<8} | "
              f"{config['precision']:<12} | {config['faux_positifs']:<10}")
    
    print("="*80)

# ============================================================
# Conseils d'Ajustement
# ============================================================
def conseils_ajustement():
    """Affiche des conseils pour l'ajustement"""
    print("\n" + "="*80)
    print("  CONSEILS D'AJUSTEMENT")
    print("="*80)
    
    print("\n📊 Quand augmenter n_components (20 → 100)?")
    print("  ✓ Si la précision n'est pas suffisante")
    print("  ✓ Si beaucoup de faux positifs")
    print("  ✗ Si la vitesse est importante")
    print("  ⚠️  Au-delà de 100, risque de surapprentissage")
    
    print("\n📍 Quand diminuer le threshold (7000 → 3000)?")
    print("  ✓ Si trop de visages sont reconnus comme inconnus")
    print("  ✓ Pour être plus strict")
    print("  ✗ Si augmentation des faux positifs")
    
    print("\n📍 Quand augmenter le threshold (3000 → 7000)?")
    print("  ✓ Si trop peu de visages sont reconnus")
    print("  ✓ Pour être plus permissif")
    print("  ✗ Si augmentation des faux positifs")
    
    print("\n🎯 Configuration Recommandée:")
    print("  • n_components: 50 (bon compromis)")
    print("  • threshold: 5000 (valeur neutre)")
    print("  • Pour ajuster: testez et observez les résultats")
    
    print("\n⚡ Si vous avez besoin de vitesse:")
    print("  • Diminuez n_components à 20-30")
    print("  • Caméra: 25-30 FPS")
    
    print("\n🎯 Si vous avez besoin de précision:")
    print("  • Augmentez n_components à 80-100")
    print("  • Caméra: 15-20 FPS")
    
    print("="*80)

if __name__ == "__main__":

    afficher_comparaison()
    conseils_ajustement()
    while menu():
        pass

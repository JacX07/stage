import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 1. Votre fichier
chemin_poids = 'model2.pth' # N'oubliez pas de remettre le bon nom
poids_dict = torch.load(chemin_poids, map_location='cpu')


# 2. Afficher tous les noms des couches pour savoir quoi visualiser
print("--- Voici les couches disponibles dans votre modèle ---")
# On n'affiche que les 20 premières pour ne pas spammer la console, enlevez le [:20] pour tout voir
for nom_couche in list(poids_dict.keys()): 
    print(f"{nom_couche} : {poids_dict[nom_couche].shape}")
print("-----------------------------------------------------")

# 2. La couche que vous voulez voir
nom_couche_cible = '_orig_mod.model.inner.L_level.layers.0.mlp_t.down_proj.weight'

if nom_couche_cible in poids_dict:
    poids_couche = poids_dict[nom_couche_cible].numpy()

    if len(poids_couche.shape) == 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(poids_couche, cmap='coolwarm', center=0)
        plt.title(f"Visualisation : {nom_couche_cible}")
        plt.xlabel("Neurones d'entrée")
        plt.ylabel("Neurones de sortie")
        
        # --- LA SOLUTION EST ICI ---
        # Au lieu de plt.show(), on sauvegarde l'image :
        nom_image = "images/puzzle_weight.png"
        plt.savefig(nom_image, bbox_inches='tight')
        print(f"Succès ! L'image a été sauvegardée sous le nom '{nom_image}' dans votre dossier actuel.")
        
    else:
        print(f"La couche a {len(poids_couche.shape)} dimensions. La Heatmap nécessite de la 2D.")
else:
    print(f"Erreur : La couche '{nom_couche_cible}' n'existe pas.")
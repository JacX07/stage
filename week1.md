Tâches effectuées cette semaine:
- première lecture et compréhension de l'article
- effecuter une première expérience avec le github de la chercheuse avec paramètres modifiés pour tester: 500 epochs pour 100 exemples et 100 exemples supplémentaires par exemples avec data augmentation.
- entrainement du modèle sur le benchmark sudoku 
- ajout d'un evaluator pour le benchmark sudoku (absent dans le dépot de base uniquement présent pour arc agi)
- modification de la boucle d'entrainement pour ne pas évaluer à chaque epochs
- résultat sur les données entrainements: ~10%
- résultats sur les données de tests: 0% à revoir car pas assez de données et entrainement trop court, mettre + d'epochs

Pistes de reflexion, à faire plus tard:
- utiliser les deux gpu 48go du serveur pour effectuer un entrainement plus conséquent 30 heures et obtenir les résultats du papier sur le benchmark sudoku ~87% avec des couches MLP.
- modifier les params pour voir l'impact sur les résultats
- réflechir sur le nombre d'étapes de récursion à effectuer pour l'instant 16 lors de l'inférence
- lecture de papiers de chercheurs s'étant inspiré du papier de Alexia Jolicoeur
- compréhension plus fine du modèle et des couches ainsi que de l'étape de récursion
- noter mes modifications
- modèle génératif?
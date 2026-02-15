Lundi 9 février:

- Début entrainement du modèle avec les paramètres du papier sur le benchmark sudoku extreme
- Compréhension du modèle recursif 
- Lecture de papiers de recherche

Mardi 10 février:

- Fin de l'entrainement ~13h sur benchmark sudoku extreme sans self attention, MLP seulement
- 87.59% de précision sur le dataset de test ce qui est similaire aux résultats du papier 
- 5 millions de paramètres

![alt text](image.png)

- Tester le modèle sur le benchmark maze
- s'intéresser à l'implémentation de lucidrains
- J'ai obtenu des résultats intéressants en entrainant le modèle sur la résolution de maze hard, en testant sur le dataset sudoku, j'obtiens quand même 21% d'accuracy pour 500 epochs , signifiant que le modèle a appris une logique qui peut se transférer sur d'autres puzzles mais de manière imparfaite.

mercredi 11 février:

- Test du modele sur le benchmark maze hard avec un gpu entrainement de 60h

Jeudi 12 février: 

- Entrainement du trm sur le dataset maze hard et test sur données sudoku, on obtient 0%, il peut s'agir d'un bug par rapport aux résultats obtenus mardi

Vendredi 13 février:

- résultats entrainement du trm sur dataset maze hard avec 50000 epochs on a des résultats mitigé sur la résolution du puzzle:
![alt text](image-2.png)

- Tache difficile, et pas de résultats de la chercheuse avec quoi comparer.
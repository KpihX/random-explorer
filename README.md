# INF421 — Path Planning Algorithms — Random Explorer

**Auteurs**

- Pacifique000
- KpihX

## Description

Ce projet est développé dans le cadre du cours INF421. Il fournit des outils pour analyser, valider et visualiser des environnements de planification de chemin (Path Planning) à partir de fichiers de scénarios textuels.

L'application permet de parser des configurations d'obstacles, de points de départ et d'arrivée, et de les représenter graphiquement pour tester des algorithmes de recherche de chemin (comme RRT).

## Installation

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion des dépendances.

### Prérequis

- Python 3.11 ou supérieur
- `uv`

### Configuration de l'environnement

```bash
# Installation des dépendances et de l'environnement virtuel
uv sync
```

## Utilisation

Le projet expose une interface en ligne de commande (CLI) avec plusieurs commandes.

### Visualiser un environnement

Pour afficher graphiquement l'environnement défini dans un fichier texte :

```bash
uv run plot-environment 
```

Cela générera une fenêtre matplotlib affichant les limites, la grille, les obstacles, ainsi que les points de départ et d'arrivée.

### Valider un fichier d'entrée

Pour vérifier si un fichier de scénario est correctement formaté :

```bash
uv run valid-input
```

## Format des Données (Fichiers Scénarios)

Les fichiers d'entrée (ex: `data/scenario0.txt`) suivent une structure stricte ligne par ligne :

| Ligne(s) | Variable   | Description                                                         |
| -------- | ---------- | ------------------------------------------------------------------- |
| 1        | `xmax`   | Limite maximale en X de l'environnement                             |
| 2        | `ymax`   | Limite maximale en Y de l'environnement                             |
| 3        | `u_s1.x` | Coordonnée X du point de départ 1                                 |
| 4        | `u_s1.y` | Coordonnée Y du point de départ 1                                 |
| 5        | `u_d1.x` | Coordonnée X de la destination 1                                   |
| 6        | `u_d1.y` | Coordonnée Y de la destination 1                                   |
| 7        | `u_s2.x` | Coordonnée X du point de départ 2                                 |
| 8        | `u_s2.y` | Coordonnée Y du point de départ 2                                 |
| 9        | `u_d2.x` | Coordonnée X de la destination 2                                   |
| 10       | `u_d2.y` | Coordonnée Y de la destination 2                                   |
| 11       | `R`      | Paramètre de rayon ou pas                                          |
| 12+      | `Obs`    | Obstacles. Chaque ligne contient 4 valeurs :`x y largeur hauteur` |

**Exemple de ligne d'obstacle :**

```text
10.0 15.0 5.0 5.0
```

Définit un rectangle à la position (10, 15) de largeur 5 et hauteur 5.

## Documentation

Pour plus de détails théoriques et complets sur le projet, veuillez vous référer au fichier `Random_Explorer.pdf`.

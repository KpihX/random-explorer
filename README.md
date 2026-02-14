# Random Explorer — INF421 Path Planning Algorithms

**Auteurs** : KpihX & Pacifique000
**École Polytechnique (l'X)** — Cours INF421

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Description

Ce projet implémente et compare des algorithmes de planification de chemin (**Path Planning**) dans des environnements 2D statiques avec obstacles. Il a été développé dans le cadre des Travaux Pratiques du cours INF421.

Il se concentre sur deux familles d'algorithmes :
1.  **PSO (Particle Swarm Optimization)** et ses variantes avancées.
2.  **RRT\* (Rapidly-exploring Random Tree Star)** pour la recherche basée sur l'échantillonnage.

Le projet inclut une visualisation riche, un système de benchmarking complet, et une optimisation d'hyperparamètres via Grid Search.

## 🚀 Fonctionnalités Clés

### 🧠 Algorithmes Implémentés
*   **PSO Basic** : Optimisation par essaim particulaire standard pour minimiser la longueur du chemin et les collisions.
*   **PSO Variants** (Architecture modulaire) :
    *   `PSORestart` : Réinitialisation périodique pour échapper aux optima locaux.
    *   `PSOSimulatedAnnealing` (SA) : Acceptation probabiliste de solutions dégradées (Metropolis criterion).
    *   `PSODimensionalLearning` (DL) : Apprentissage dimension par dimension pour les particules stagnantes (Xu et al., 2019).
    *   `PSOAdaptiveInertia` : Poids d'inertie linéairement décroissant + Early Stopping.
*   **RRT\*** : Version optimisée asymptotiquement optimale du RRT avec rewiring.
*   **Multi-Robot RRT\*** : Extension pour la planification de chemin avec plusieurs robots.

### 🛠️ Outils & Infrastructure
*   **Visualisation** : Plotting via `matplotlib` des environnements, obstacles, chemins et arbres de recherche.
*   **Benchmarks** : Comparaison automatisée (temps, longueur, taux de succès) entre algorithmes.
*   **Grid Search** : Script parallélisé (`ProcessPoolExecutor`) pour tuner les hyperparamètres.
*   **Robustesse** : Gestion des collisions Soft (pénalité proportionnelle) et Hard.
*   **Interface** : CLI moderne avec `typer` et feedback visuel avec `rich` et `tqdm`.

## 📂 Structure du Projet

```bash
random-explorer/
├── data/                       # Scénarios de test (.txt) et résultats JSON
├── src/random_explorer/
│   ├── environment.py          # Parsing, collisions (Liang-Barsky), affichage
│   ├── benchmark.py            # Moteur de benchmark
│   ├── pso/                    # Package des variantes PSO
│   │   ├── path_planner.py     # PSO Base
│   │   ├── restart.py          # + Random Restart
│   │   ├── simulated_annealing.py
│   │   ├── dimensional_learning.py
│   │   └── adaptive_inertia.py
│   ├── rrt/                    # Package des algorithmes RRT*
│   │   ├── node.py             # Structure de nœud pour l'arbre
│   │   ├── planner.py          # RRT* principal
│   │   └── multi_robot.py      # Multi-Robot RRT*
│   └── scripts/                # Entry points CLI
├── Random_Explorer.ipynb       # Rapport exécutable (Notebook Jupyter)
├── pyproject.toml              # Configuration et dépendances
└── README.md
```

## 📦 Installation

Ce projet utilise [uv](https://github.com/astral-sh/uv) pour une gestion ultra-rapide des dépendances.

1.  **Cloner le dépôt**
    ```bash
    git clone https://github.com/KpihX/random-explorer.git
    cd random-explorer
    ```

2.  **Installer l'environnement**
    ```bash
    uv sync
    ```

## 🎮 Utilisation

### Via le CLI (Ligne de Commande)

Le projet expose plusieurs commandes via `uv run` :

**1. Visualiser un scénario**
Affiche l'environnement, le départ, l'arrivée et les obstacles.
```bash
uv run plot-environment --file data/scenario0.txt
```

**2. Valider un fichier d'entrée**
Vérifie le format et les contraintes d'un fichier scénario.
```bash
uv run valid-input --file data/scenario0.txt
```

**3. Lancer un Grid Search PSO**
Lance une recherche d'hyperparamètres parallélisée pour optimiser les variantes PSO.
```bash
uv run grid-search --scenario data/scenario2.txt --workers 8
```

### Via le Notebook Jupyter

Le fichier `Random_Explorer.ipynb` est le point d'entrée principal pour explorer les réponses aux questions du TP. Il contient :
*   Les explications théoriques (avec formules LaTeX).
*   L'exécution pas-à-pas des algorithmes.
*   Les courbes de convergence et les comparaisons visuelles.

Pour le lancer :
```bash
uv run jupyter lab Random_Explorer.ipynb
```

## 📊 Format des Données (Scénarios)

Les fichiers `.txt` dans `data/` suivent ce format strict :

| Ligne | Contenu                | Description                             |
| ----- | ---------------------- | --------------------------------------- |
| 1-2   | `width`, `height`      | Dimensions de l'environnement           |
| 3-4   | `start1_x`, `start1_y` | Point de départ 1 (utilisé par PSO/RRT) |
| 5-6   | `goal1_x`, `goal1_y`   | Objectif 1                              |
| 7-10  | ...                    | Points pour Robot 2 (optionnel)         |
| 11    | `R`                    | Rayon de sécurité                       |
| 12+   | `x y w h`              | Liste des obstacles (rectangles)        |

## 🧪 Résultats & Performance

Les résultats des benchmarks montrent que :
*   **RRT\*** est généralement plus robuste et garantit (probabilistiquement) de trouver un chemin s'il existe.
*   **PSO** est très rapide sur des environnements simples mais nécessite un tuning fin (d'où l'importance du Grid Search et des variantes comme *Adaptive Inertia*).
*   L'implémentation **vectorisée** (numpy) assure de bonnes performances même avec de nombreuses particules.

---
*Projet réalisé pour le cours INF421 - 2025/2026*

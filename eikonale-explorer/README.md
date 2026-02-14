# Eikonale Explorer — Path Planning via Eikonal Equation

**Auteurs** : [KpihX](https://github.com/KpihX) & [Pacifique000](https://github.com/Pacifique000)  
**École Polytechnique (l'X)** — Cours INF421 TP7 (Refactoring Complet)

---

## 🎯 Objectif

Ce projet propose une approche de **planification de chemin optimal** basée sur la résolution numérique de l'**Équation Eikonale**. Il transforme un problème de navigation en un problème de propagation de front d'onde.

$$|\nabla \phi(x)| = N(x)$$

où $\phi(x)$ représente le temps d'arrivée et $N(x)$ l'indice de réfraction local (coût de traversée).

## 🚀 Fonctionnalités

| Composant                    | Description                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| **Solveur Lax-Friedrichs**   | Schéma numérique itératif pour résoudre l'équation Eikonale       |
| **Reconstruction de Chemin** | Descente de gradient (Euler, Heun/RK2)                            |
| **Fonctions d'Indice**       | Uniforme, dioptre, gaussien, obstacles, île                       |
| **Environnement**            | Parsing des scénarios `.txt` (compatibles avec `random-explorer`) |
| **CLI**                      | Interface ligne de commande avec Typer                            |
| **Visualisation**            | Contours, gradients, chemins                                      |

## 📂 Structure du Projet

```
eikonale-explorer/
├── data/                           # Scénarios de test (.txt)
├── src/eikonale_explorer/
│   ├── __init__.py                 # Exports du package
│   ├── environment.py              # Parsing scénarios
│   ├── index_functions.py          # Générateurs de cartes N(x,y)
│   ├── path_finder.py              # Euler, Heun, interpolation
│   ├── plotting.py                 # Visualisation
│   ├── config.py / config.yaml     # Configuration
│   ├── utils.py                    # Console Rich
│   ├── solvers/
│   │   ├── base.py                 # Classe abstraite
│   │   └── lax_friedrichs.py       # Implémentation LF
│   └── scripts/
│       ├── cli.py                  # CLI principal
│       ├── solve.py                # Pipeline complet
│       └── plot_environment.py     # Visualisation env
├── Eikonale_Explorer.ipynb         # Notebook de démonstration
├── pyproject.toml                  # Dépendances (uv)
└── README.md
```

## 📦 Installation

### Avec [uv](https://github.com/astral-sh/uv) (recommandé)

```bash
cd eikonale-explorer
uv sync
```

### Depuis GitHub

```bash
uv pip install git+https://github.com/KpihX/random-explorer.git#subdirectory=eikonale-explorer
```

## 🎮 Utilisation

### CLI

**Résoudre un scénario complet :**

```bash
uv run eikonal-explorer solve --file data/scenario0.txt --grid-size 200 --max-iter 5000
```

**Visualiser l'environnement :**

```bash
uv run eikonal-plot --file data/scenario0.txt
```

### Notebook

Le notebook `Eikonale_Explorer.ipynb` offre une exploration interactive complète :

```bash
uv run jupyter lab Eikonale_Explorer.ipynb
```

**Contenu du notebook :**

1. **Partie 1** : Résolution Eikonale (cas uniforme, dioptre, gaussien)
2. **Partie 2** : Reconstruction de chemins (Euler vs Heun)
3. **Partie 3** : Obstacles et terrains complexes
4. **Partie 4** : L'île avec différents terrains (forêt, montagne, rivière)
5. **Partie 5** : Application aux scénarios `random-explorer`

### API Python

```python
from eikonale_explorer import (
    Environment,
    LaxFriedrichsSolver,
    PathFinder,
    create_diopter_index,
    IndexConfig,
    plot_contour_lines,
)

# Charger un scénario
env = Environment("data/scenario0.txt")

# Générer la carte de coûts
config = IndexConfig(nx=128, ny=128)
N_map = env.get_refractive_index_map(128, 128, obstacle_index=1e5)

# Résoudre l'équation Eikonale
solver = LaxFriedrichsSolver(max_iter=3000, tol=1e-6)
phi = solver.solve(N_map, source_norm, h)

# Reconstruire le chemin optimal
path = PathFinder.solve_heun(phi, source_norm, goal_norm, h)
```

## 🧠 Théorie

### Équation Eikonale

L'équation Eikonale modélise la propagation d'un front d'onde :

$$|\nabla \phi| = N(x,y)$$

- $\phi(x,y)$ : temps d'arrivée (distance temporelle depuis la source)
- $N(x,y)$ : indice de réfraction (inverse de la vitesse locale)
- $N = 1$ : espace libre
- $N \gg 1$ : obstacle (vitesse quasi-nulle)

### Schéma de Lax-Friedrichs

Itération numérique pour résoudre l'EDP :

$$\phi_{i,j}^{n+1} = \frac{\phi_{i+1,j} + \phi_{i-1,j} + \phi_{i,j+1} + \phi_{i,j-1}}{4} - \Delta t \cdot H(\nabla\phi, N)$$

avec $H(\nabla\phi, N) = |\nabla\phi| - N$ le Hamiltonien.

### Reconstruction de Chemin

Une fois $\phi$ calculé, le chemin optimal suit la descente de gradient :

$$\frac{dX}{dt} = -\frac{\nabla\phi}{|\nabla\phi|}$$

résolue par Euler ou Heun (RK2) depuis la destination vers la source.

## 📊 Exemple de Résultats

| Cas                 | Description                               |
| ------------------- | ----------------------------------------- |
| N=1 (uniforme)      | Lignes de niveau circulaires              |
| Dioptre (mer/plage) | Réfraction à l'interface (loi de Snell)   |
| Obstacle            | Contournement automatique                 |
| Île                 | Chemin évitant rivière, utilisant le pont |

## 🔗 Liens

- **Projet parent** : [random-explorer](https://github.com/KpihX/random-explorer) (PSO, RRT*)
- **Données** : [Kaggle Dataset](https://kaggle.com/datasets/ivannkamdem/random-explorer)
- **Documentation INF421** : TP7 - Équation Eikonale

---

*Projet réalisé dans le cadre du cours INF421 à l'École Polytechnique.*

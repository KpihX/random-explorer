# Eikonale Explorer — Path Planning via Eikonal Equation

**Auteurs** : KpihX & Pacifique000  
**École Polytechnique (l'X)** — Cours INF421 TP7 (Refactoring)

Ce projet propose une approche de planification de chemin basée sur la résolution numérique de l'**Équation Eikonale**. Il transforme un problème de recherche de chemin en un problème de propagation de front d'onde dans un milieu à indice de réfraction variable.

## 🚀 Fonctionnalités

*   **Solveur Eikonal** : Implémentation du schéma numérique de **Lax-Friedrichs** pour résoudre $|\nabla \phi| = N$.
*   **Reconstruction de Chemin** : Méthodes de descente de gradient (**Euler** et **Heun**) sur le champ de potentiel $\phi$.
*   **Environnement** : Parsing des fichiers de scénarios (`.txt`) et génération automatique de cartes de coûts (indices de réfraction).
*   **Visualisation** : Outils CLI et Notebook pour visualiser les iso-contours (lignes de niveau) et les trajectoires optimales.

## 📂 Structure

Le projet est structuré comme un package Python moderne, inspiré de `random-explorer` :

```bash
eikonale-explorer/
├── data/                       # Scénarios de test (.txt)
├── src/eikonale_explorer/
│   ├── environment.py          # Gestion de la carte et des obstacles
│   ├── solvers/                # Algorithmes de résolution EDP (Lax-Friedrichs)
│   ├── path_finder.py          # Reconstruction de chemin (Euler/Heun)
│   └── scripts/                # Interface CLI (solve, plot)
├── Eikonale_Explorer.ipynb     # Notebook de démonstration
├── pyproject.toml              # Configuration et dépendances
└── README.md
```

## 📦 Installation

Ce projet utilise [uv](https://github.com/astral-sh/uv).

```bash
cd eikonale-explorer
uv sync
```

## 🎮 Utilisation

### Ligne de Commande (CLI)

**Résoudre et visualiser un scénario :**

```bash
uv run eikonal-solve --file data/scenario0.txt --grid-size 128 --max-iter 3000
```
Cela affichera une fenêtre avec les obstacles, les contours du champ Eikonal et le chemin optimal trouvé.

**Visualiser l'environnement seul :**

```bash
uv run eikonal-plot --file data/scenario0.txt
```

### Notebook

Ouvrez `Eikonale_Explorer.ipynb` avec Jupyter pour une exploration interactive :

```bash
uv run jupyter lab Eikonale_Explorer.ipynb
```

## 🧠 Théorie

L'équation Eikonale :
$$ |\nabla \phi(x)| = N(x) $$
avec la condition aux limites $\phi(x_{start}) = 0$.

$N(x)$ est l'indice de réfraction :
*   $N(x) = 1$ dans l'espace vide.
*   $N(x) \gg 1$ (coût élevé) dans les obstacles.

# Copilot Instructions for random-explorer

## Behavioral Rules (Strict)
1.  **Plan First**: For every request, ALWAYS present a detailed plan with the "whys" and "hows".
2.  **Wait for Approval**: NEVER modify any code or file until the user explicitly gives the "green light" (le feu vert).
3.  **Conform to Practices**: Strictly follow the existing code style, project structure (src/random_explorer layout), and conventions (uv, typer, rich).
4.  **Scoped Modifications**: Only modify exactly what is related to the user's question. Do not refactor, "clean up", or fix unrelated issues unless explicitly asked.

## Project Overview
-   **Domain**: Path Planning Algorithms (INF421) - RRT, RRT*, etc.
-   **Goal**: Parse scenario descriptions and visualize path planning environments.
-   **Package Manager**: `uv`. Use `uv` for all dependency and execution tasks.

## Architecture & Code Structure
-   **CLI Application**: `src/random_explorer/cli.py` uses `typer` to expose commands.
-   **Data Parsing**: `src/random_explorer/valid_input.py` parses the strict line-based `.txt` scenario format.
-   **Visualization**: `src/random_explorer/plot_environment.py` renders the environment using `matplotlib`.
-   **Configuration**: `src/random_explorer/config.py` loads settings from `config.yaml` using `lru_cache`.

## Critical Workflows
-   **Run Visualization**:
    ```bash
    uv run plot-environment -f data/scenario0.txt
    ```
-   **Validate Input**:
    ```bash
    uv run valid-input -f data/scenario0.txt
    ```
-   **Install Dependencies**:
    ```bash
    uv sync
    ```

## Data Format (Scenario Files)
Input `.txt` files in `data/` must follow this structure (parsed by `valid_input.py`):
-   **Line 1**: `xmax` (Environment X limit)
-   **Line 2**: `ymax` (Environment Y limit)
-   **Line 3-4**: Start Point 1 (`u_s1`: x, y)
-   **Line 5-6**: Destination 1 (`u_d1`: x, y)
-   **Line 7-8**: Start Point 2 (`u_s2`: x, y)
-   **Line 9-10**: Destination 2 (`u_d2`: x, y)
-   **Line 11**: Parameter `R` (Radius/Stepsize)
-   **Line 12+**: Obstacles, each line containing 4 floats: `x, y, width, height` (used for `patches.Rectangle`).

# Copilot Instructions for random-explorer

## Behavioral Rules (Strict)
1.  **Plan First**: For every request, ALWAYS present a detailed plan with the "whys" and "hows".
2.  **Wait for Approval**: NEVER modify any code or file until the user explicitly gives the "green light" (le feu vert).
3.  **Conform to Practices**: Strictly follow the existing code style, project structure (src/random_explorer layout), and conventions (uv, typer, rich).
4.  **Scoped Modifications**: Only modify exactly what is related to the user's question. Do not refactor, "clean up", or fix unrelated issues unless explicitly asked.

**Note**: This is a collaborative project. We work together step by step.

## Code Philosophy (Critical)

### No Hardcoding
- All constants externalized in `config.yaml`
- Use `config.py` with `lru_cache` to load configurations
- Never hardcode file paths, parameters, or magic numbers

### Modularity First
- POO approach: autonomous classes with proper encapsulation
- Break code into independent modules/packages
- Avoid spaghetti code
- Each class manages itself autonomously

### Transparency
- Always provide logs/verbose mode (using Rich)
- Facilitate debugging and internal machinery comprehension
- Display clear progress and error messages

### Terminal UI Stack
- **CLI**: Typer for command-line interface
- **Display**: Rich (Rich-Json, Rich-Markdown, Panels, Pretty)
- **Console**: Custom `Console` class in `utils.py` extending `RichConsole`

### Package Manager: 100% uv
- **NEVER use**: hatch, pip, python, pipx directly
- **ALWAYS use**: `uv add`, `uv sync`, `uv run`, `uv tool install`
- **Builder**: `uv_build` (not hatch)
- All dependency management through `pyproject.toml` + `uv.lock`

### DRY Principle (Don't Repeat Yourself)
- Use `utils.py` for shared functions/definitions
- Maintain holistic vision before coding
- Check if code already exists elsewhere before rewriting
- Externalize common code and import it
- Think about reusability from the start

## Project Overview
-   **Domain**: Path Planning Algorithms (INF421 - Academic Project)
-   **Algorithms**: RRT (Rapidly-exploring Random Tree), PSO (Particle Swarm Optimization)
-   **Goal**: Parse, validate, and visualize 2-robot path planning scenarios
-   **Project Structure**: 25 questions covering implementation and optimization
-   **Data Source**: Kaggle dataset (`ivannkamdem/random-explorer`)

## Architecture & Code Structure
-   **CLI**: `src/random_explorer/cli.py` (Typer-based commands)
-   **Parsing**: `src/random_explorer/valid_input.py` (strict line-based format)
-   **Visualization**: `src/random_explorer/plot_environment.py` (matplotlib rendering)
-   **Configuration**: `src/random_explorer/config.py` + `config.yaml` (lru_cache pattern)
-   **Utilities**: `src/random_explorer/utils.py` (Console class, shared functions)

## Critical Workflows

### Installation
```bash
uv sync
```

### Validation
```bash
# Uses default file from config.yaml
uv run valid-input

# Specify custom file
uv run valid-input -f data/scenario0.txt
```

### Visualization
```bash
# Uses default file from config.yaml
uv run plot-environment

# Specify custom file
uv run plot-environment -f data/scenario1.txt
```

### Python Module Execution (Alternative)
```bash
uv run python -m random_explorer.valid_input
uv run python -m random_explorer.plot_environment
```

## Data Format (Scenario Files)

Two-robot path planning environment. Input `.txt` files in `data/` follow this structure:

-   **Line 1**: `xmax` (Environment X boundary)
-   **Line 2**: `ymax` (Environment Y boundary)
-   **Line 3-4**: Robot 1 start position (`u_s1`: x, y)
-   **Line 5-6**: Robot 1 destination (`u_d1`: x, y)
-   **Line 7-8**: Robot 2 start position (`u_s2`: x, y)
-   **Line 9-10**: Robot 2 destination (`u_d2`: x, y)
-   **Line 11**: `R` (Safety zone radius around each robot)
-   **Line 12+**: Obstacles (format: `x y width height` per line, space-separated)

**Note**: Values are in scientific notation (e.g., `1.0000000e+03` = 1000). Obstacles are rectangles rendered with `matplotlib.patches.Rectangle`.

"""Grid Search for PSO hyperparameter optimization.

This script performs grid search cross-validation to find optimal
hyperparameters for PSO path planners and variants.

Supports: PSOPathPlanner, PSORestart, PSOSimulatedAnnealing, 
          PSODimensionalLearning, PSOAdaptiveInertia

Usage:
    random-explorer grid-search --scenarios "4" --variant basic
    random-explorer grid-search --scenarios "4" --variant restart
    random-explorer grid-search --scenarios "4" --variant sa
    random-explorer grid-search --scenarios "4" --variant dl
    random-explorer grid-search --scenarios "4" --variant adaptive
"""

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..config import load_config
from ..environment import Environment
from ..pso import (
    PSOPathPlanner,
    PSORestart,
    PSOSimulatedAnnealing,
    PSODimensionalLearning,
    PSOAdaptiveInertia,
)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


console = Console()

# Global cache for environments
_env_cache: Dict[int, Environment] = {}

# Variant registry
VARIANTS = {
    'basic': PSOPathPlanner,
    'restart': PSORestart,
    'sa': PSOSimulatedAnnealing,
    'dl': PSODimensionalLearning,
    'adaptive': PSOAdaptiveInertia,
}

# Base parameters (common to all variants)
BASE_PARAM_GRID = {
    'num_particles': [250, 300, 400],
    'num_waypoints': [8],
    'max_iter': [100, 200],
    'w': [0.5, 0.7, 0.85],
    'c1': [0.8, 1.0, 1.2, 1.4, 1.6],
    'c2': [0.8, 1.0, 1.2, 1.4, 1.6],
}

# Variant-specific parameters
VARIANT_PARAM_GRIDS = {
    'basic': {},  # No extra params
    'restart': {
        'restart_frequency': [30, 50],
        'elite_ratio': [0.7, 0.9],
    },
    'sa': {
        'T0': [50.0, 100.0],
        'beta': [0.7, 0.9],
        'restart_frequency': [200],
        'elite_ratio': [0],
    },
    'dl': {
        'wait_limit': [50, 100],
        'T0': [0],
        'beta': [0],
        'restart_frequency': [200],
        'elite_ratio': [0],
    },
    'adaptive': {
        'w_max': [0.9],
        'w_min': [0.4],
        'wait_limit': [50, 100],
        'T0': [0],
        'beta': [0],
        'restart_frequency': [200],
        'elite_ratio': [0],
    },
}

# Solve kwargs per variant (what to pass to solve())
VARIANT_SOLVE_KWARGS = {
    'basic': {},
    'restart': {},
    'sa': {'restart': False},
    'dl': {'simulated_annealing': False, 'restart': False},
    'adaptive': {'simulated_annealing': False, 'restart': False, 'dimensional_learning': False},
}


@dataclass
class GridSearchResult:
    """Result of a single parameter configuration evaluation."""
    params: Dict[str, Any]
    success_rate: float
    avg_path_length: float
    avg_score: float
    avg_time: float
    avg_iterations: float
    num_runs: int
    num_successes: int
    scenarios_tested: List[int]
    variant: str = 'basic'
    
    def __lt__(self, other: 'GridSearchResult') -> bool:
        if self.success_rate != other.success_rate:
            return self.success_rate > other.success_rate
        return self.avg_path_length < other.avg_path_length
    
    def is_better_than(self, other: Optional['GridSearchResult']) -> bool:
        if other is None:
            return True
        if self.success_rate > other.success_rate:
            return True
        if self.success_rate == other.success_rate and self.avg_path_length < other.avg_path_length:
            return True
        return False


@dataclass
class GridSearchConfig:
    """Configuration for grid search."""
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: BASE_PARAM_GRID.copy())
    scenarios: List[int] = field(default_factory=lambda: [4])
    runs_per_config: int = 3
    soft_mode: bool = True
    n_workers: int = field(default_factory=lambda: min(cpu_count(), 22))
    variant: str = 'basic'
    planner_class: Type[PSOPathPlanner] = PSOPathPlanner
    solve_kwargs: Dict[str, Any] = field(default_factory=dict)


def load_scenario(scenario_id: int) -> Environment:
    """Load a scenario environment with caching."""
    global _env_cache
    
    if scenario_id in _env_cache:
        return _env_cache[scenario_id]
    
    if HAS_KAGGLEHUB:
        file_path = kagglehub.dataset_download(
            handle="ivannkamdem/random-explorer",
            path=f"scenario{scenario_id}.txt"
        )
    else:
        file_path = Path(__file__).parent.parent.parent.parent / "data" / f"scenario{scenario_id}.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"Scenario {scenario_id} not found.")
    
    env = Environment(str(file_path))
    _env_cache[scenario_id] = env
    return env


def format_params(params: Dict[str, Any], variant: str = 'basic') -> str:
    """Format parameters as a compact string."""
    base = f"S={params['num_particles']}, N={params['num_waypoints']}, iter={params['max_iter']}"
    
    if 'w' in params:
        base += f", w={params['w']:.2f}"
    if 'w_max' in params:
        base += f", w=[{params.get('w_min', 0.4):.1f}-{params['w_max']:.1f}]"
    
    base += f", c1={params['c1']:.2f}, c2={params['c2']:.2f}"
    
    # Add variant-specific params
    if variant == 'restart' and 'restart_frequency' in params:
        base += f", rf={params['restart_frequency']}"
    if variant in ['sa', 'dl', 'adaptive'] and 'T0' in params:
        base += f", T0={params['T0']:.0f}"
    if variant in ['dl', 'adaptive'] and 'wait_limit' in params:
        base += f", wl={params['wait_limit']}"
    
    return base


def evaluate_single_run(args: Tuple) -> Tuple[bool, float, float, float, int]:
    """Evaluate a single PSO run (worker function)."""
    params, scenario_id, soft_mode, planner_class_name, solve_kwargs = args
    
    # Get planner class from name
    planner_class = VARIANTS.get(planner_class_name, PSOPathPlanner)
    if planner_class_name == 'adaptive':
        planner_class = PSOAdaptiveInertia
    elif planner_class_name == 'dl':
        planner_class = PSODimensionalLearning
    elif planner_class_name == 'sa':
        planner_class = PSOSimulatedAnnealing
    elif planner_class_name == 'restart':
        planner_class = PSORestart
    else:
        planner_class = PSOPathPlanner
    
    env = load_scenario(scenario_id)
    
    # For adaptive, remove 'w' from params
    planner_params = params.copy()
    if planner_class_name == 'adaptive' and 'w' in planner_params:
        del planner_params['w']
    
    planner = planner_class(env, **planner_params)
    
    start_time = time.time()
    path, length, score, history = planner.solve(soft_mode=soft_mode, **solve_kwargs)
    elapsed = time.time() - start_time
    
    is_valid = np.isclose(score, length)
    return is_valid, length, score, elapsed, len(history)


def evaluate_config_parallel(
    params: Dict[str, Any],
    scenarios: List[int],
    runs_per_config: int,
    soft_mode: bool,
    executor: ProcessPoolExecutor,
    variant: str = 'basic',
    solve_kwargs: Dict[str, Any] = None
) -> GridSearchResult:
    """Evaluate a parameter configuration using parallel runs."""
    if solve_kwargs is None:
        solve_kwargs = {}
    
    tasks = []
    for scenario_id in scenarios:
        for _ in range(runs_per_config):
            tasks.append((params.copy(), scenario_id, soft_mode, variant, solve_kwargs))
    
    futures = [executor.submit(evaluate_single_run, task) for task in tasks]
    
    successes = 0
    path_lengths = []
    scores = []
    times = []
    iterations = []
    
    for future in as_completed(futures):
        is_valid, length, score, elapsed, n_iter = future.result()
        
        scores.append(score)
        times.append(elapsed)
        iterations.append(n_iter)
        
        if is_valid:
            successes += 1
            path_lengths.append(length)
    
    total_runs = len(tasks)
    success_rate = successes / total_runs if total_runs > 0 else 0.0
    avg_length = np.mean(path_lengths) if path_lengths else float('inf')
    
    return GridSearchResult(
        params=params,
        success_rate=success_rate,
        avg_path_length=avg_length,
        avg_score=np.mean(scores),
        avg_time=np.mean(times),
        avg_iterations=np.mean(iterations),
        num_runs=total_runs,
        num_successes=successes,
        scenarios_tested=scenarios.copy(),
        variant=variant
    )


def generate_combinations(base_grid: Dict, variant_grid: Dict) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from base + variant grids."""
    combined = {**base_grid, **variant_grid}
    param_names = list(combined.keys())
    param_values = list(combined.values())
    
    return [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]


def grid_search(config: GridSearchConfig) -> List[GridSearchResult]:
    """Perform parallel grid search over parameter space."""
    combinations = generate_combinations(config.param_grid, {})
    
    total_configs = len(combinations)
    total_evals = total_configs * len(config.scenarios) * config.runs_per_config
    
    variant_name = config.variant.upper()
    console.print(Panel.fit(
        f"[bold blue]Grid Search - {variant_name}[/bold blue]\n"
        f"  Scenarios: {config.scenarios}\n"
        f"  Runs per config: {config.runs_per_config}\n"
        f"  Total combinations: {total_configs}\n"
        f"  Total evaluations: {total_evals}\n"
        f"  Workers: {config.n_workers} CPUs",
        title=f"🔍 PSO {variant_name} Search"
    ))
    
    results = []
    best_result: Optional[GridSearchResult] = None
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        for idx, params in enumerate(combinations, 1):
            console.print(f"\n[dim]━━━ Config {idx}/{total_configs} ━━━[/dim]")
            console.print(f"  [cyan]{format_params(params, config.variant)}[/cyan]")
            
            result = evaluate_config_parallel(
                params, 
                config.scenarios, 
                config.runs_per_config,
                config.soft_mode,
                executor,
                config.variant,
                config.solve_kwargs
            )
            results.append(result)
            
            rate_color = "green" if result.success_rate >= 0.8 else "yellow" if result.success_rate >= 0.5 else "red"
            console.print(f"  [bold {rate_color}]→ Success: {result.success_rate:.0%} ({result.num_successes}/{result.num_runs})[/bold {rate_color}]", end="")
            if result.avg_path_length < float('inf'):
                console.print(f" | Avg Length: {result.avg_path_length:.1f} | Time: {result.avg_time:.2f}s")
            else:
                console.print(f" | Time: {result.avg_time:.2f}s")
            
            if result.is_better_than(best_result):
                best_result = result
                console.print(Panel.fit(
                    f"[bold green]🏆 NEW BEST![/bold green]\n"
                    f"  Success: [bold]{result.success_rate:.0%}[/bold] ({result.num_successes}/{result.num_runs})\n"
                    f"  Avg Length: [bold]{result.avg_path_length:.1f}[/bold]\n"
                    f"  [cyan]{format_params(result.params, config.variant)}[/cyan]",
                    border_style="green"
                ))
    
    elapsed_total = time.time() - start_time
    console.print(f"\n[bold]Total search time: {elapsed_total:.1f}s[/bold]")
    
    results.sort()
    return results


def display_results(results: List[GridSearchResult], top_n: int = 10, variant: str = 'basic') -> None:
    """Display grid search results in a nice table."""
    console.print(f"\n[bold green]═══ Top {min(top_n, len(results))} {variant.upper()} Configurations ═══[/bold green]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Success", justify="right")
    table.add_column("Avg Length", justify="right")
    table.add_column("Avg Time", justify="right")
    table.add_column("S", justify="right", width=4)
    table.add_column("w", justify="right", width=5)
    table.add_column("c1", justify="right", width=5)
    table.add_column("c2", justify="right", width=5)
    
    # Add variant-specific columns
    if variant == 'restart':
        table.add_column("rf", justify="right", width=4)
        table.add_column("elite", justify="right", width=5)
    elif variant in ['sa', 'dl', 'adaptive']:
        table.add_column("T0", justify="right", width=5)
    if variant in ['dl', 'adaptive']:
        table.add_column("wl", justify="right", width=4)
    
    for i, r in enumerate(results[:top_n]):
        success_str = f"{r.success_rate:.0%} ({r.num_successes}/{r.num_runs})"
        length_str = f"{r.avg_path_length:.1f}" if r.avg_path_length < float('inf') else "N/A"
        
        if r.success_rate >= 1.0:
            style = "bold green"
        elif r.success_rate >= 0.8:
            style = "green"
        elif r.success_rate >= 0.5:
            style = "yellow"
        else:
            style = "red"
        
        row = [
            str(i + 1),
            success_str,
            length_str,
            f"{r.avg_time:.2f}s",
            str(r.params['num_particles']),
            f"{r.params.get('w', r.params.get('w_max', 0.9)):.2f}",
            f"{r.params['c1']:.1f}",
            f"{r.params['c2']:.1f}",
        ]
        
        if variant == 'restart':
            row.append(str(r.params.get('restart_frequency', '')))
            row.append(f"{r.params.get('elite_ratio', 0):.1f}")
        elif variant in ['sa', 'dl', 'adaptive']:
            row.append(f"{r.params.get('T0', 0):.0f}")
        if variant in ['dl', 'adaptive']:
            row.append(str(r.params.get('wait_limit', '')))
        
        table.add_row(*row, style=style)
    
    console.print(table)
    
    if results:
        best = results[0]
        console.print(Panel.fit(
            f"[bold]Success rate:[/bold] {best.success_rate:.0%}\n"
            f"[bold]Avg path length:[/bold] {best.avg_path_length:.1f}\n"
            f"[bold]Avg time:[/bold] {best.avg_time:.2f}s\n\n"
            f"[bold cyan]Parameters:[/bold cyan]\n"
            f"{json.dumps(best.params, indent=2)}",
            title=f"🏆 Best {variant.upper()} Configuration",
            border_style="green"
        ))


def main(
    scenarios: Optional[List[int]] = None,
    runs: int = 3,
    output: Optional[str] = None,
    workers: Optional[int] = None,
    variant: str = 'basic'
) -> List[GridSearchResult]:
    """Run grid search for PSO hyperparameters.
    
    Args:
        scenarios: List of scenario IDs to test.
        runs: Number of runs per configuration.
        output: Optional path to save results as JSON.
        workers: Number of parallel workers.
        variant: PSO variant ('basic', 'restart', 'sa', 'dl', 'adaptive').
    """
    if scenarios is None:
        scenarios = [4]
    
    if workers is None:
        workers = min(cpu_count(), 22)
    
    # Build param grid: base + variant-specific
    param_grid = BASE_PARAM_GRID.copy()
    variant_params = VARIANT_PARAM_GRIDS.get(variant, {})
    param_grid.update(variant_params)
    
    # For adaptive variant, replace 'w' with 'w_max'/'w_min'
    if variant == 'adaptive' and 'w' in param_grid:
        del param_grid['w']
    
    solve_kwargs = VARIANT_SOLVE_KWARGS.get(variant, {})
    
    config = GridSearchConfig(
        param_grid=param_grid,
        scenarios=scenarios,
        runs_per_config=runs,
        soft_mode=True,
        n_workers=workers,
        variant=variant,
        solve_kwargs=solve_kwargs
    )
    
    console.print(f"[bold]PSO {variant.upper()} Hyperparameter Grid Search[/bold]")
    console.print("=" * 50)
    
    results = grid_search(config)
    display_results(results, top_n=15, variant=variant)
    
    # Default output from config.yaml
    if output is None:
        config = load_config()
        output_subdir = config.get('grid_search_output_dir', 'data/grid_search') if config else 'data/grid_search'
        output_dir = PROJECT_ROOT / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output = str(output_dir / f"grid_search_{variant}.json")
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'variant': variant,
        'scenarios': scenarios,
        'runs_per_config': runs,
        'results': [
            {
                'params': r.params,
                'success_rate': r.success_rate,
                'avg_path_length': r.avg_path_length if r.avg_path_length < float('inf') else None,
                'avg_score': r.avg_score,
                'avg_time': r.avg_time,
                'num_successes': r.num_successes,
                'num_runs': r.num_runs
            }
            for r in results
        ]
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    console.print(f"\n[dim]Results saved to {output}[/dim]")
    
    return results


if __name__ == "__main__":
    import typer
    
    def cli_main(
        scenarios: str = typer.Option("4", "-s", "--scenarios", help="Scenario IDs (comma-separated)"),
        runs: int = typer.Option(3, "-r", "--runs", help="Runs per configuration"),
        output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSON file path"),
        workers: Optional[int] = typer.Option(None, "-w", "--workers", help="Number of parallel workers"),
        variant: str = typer.Option("basic", "-v", "--variant", help="PSO variant: basic, restart, sa, dl, adaptive")
    ):
        """Run grid search for PSO hyperparameters."""
        scenario_list = [int(s.strip()) for s in scenarios.split(",")]
        main(scenarios=scenario_list, runs=runs, output=output, workers=workers, variant=variant)
    
    typer.run(cli_main)

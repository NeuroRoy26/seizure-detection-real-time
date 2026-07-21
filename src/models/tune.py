"""
=============================================================================
  AUTOMATED GRID SEARCH HYPERPARAMETER TUNER
=============================================================================
Sweeps across a defined grid of learning rates and batch sizes, executes
the model training loop for each configuration, and tracks comparative
metrics and parameters directly inside MLflow/DAGsHub.
=============================================================================
"""

import os
import sys
import copy

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_model, _load_config

def run_grid_search():
    # Load base master configuration
    base_config = _load_config()
    
    # ── DEFINE HYPERPARAMETER SEARCH SPACE ──
    # Note: Keep the grid small initially for fast execution.
    learning_rates = [0.005, 0.001]
    batch_sizes = [32, 64]
    
    # For quick tuning verification, we can set epochs to a smaller value (e.g., 3)
    # Set this to base_config["training"]["epochs"] if you want to run the full training.
    epochs_to_run = 3
    
    total_runs = len(learning_rates) * len(batch_sizes)
    print("=" * 60)
    
    print(f"🚀 INITIALIZING HYPERPARAMETER TUNING | TOTAL RUNS: {total_runs}")
    print("=" * 60)
    print(f"  Sweep Grid:")
    print(f"    - Learning Rates: {learning_rates}")
    print(f"    - Batch Sizes   : {batch_sizes}")
    print(f"    - Epochs per run: {epochs_to_run}\n")

    run_count = 0
    for lr in learning_rates:
        for bs in batch_sizes:
            run_count += 1
            print("\n" + "#" * 60)
            print(f" RUN {run_count}/{total_runs}: LR = {lr} | BATCH SIZE = {bs}")
            print("#" * 60)
            
            # Deep copy base config to avoid leaking overrides between runs
            run_config = copy.deepcopy(base_config)
            
            # Apply overrides for this grid point
            run_config["training"]["learning_rate"] = lr
            run_config["training"]["batch_size"] = bs
            run_config["training"]["epochs"] = epochs_to_run
            
            try:
                # Trigger training loop
                train_model(run_config)
                print(f"[SUCCESS] Finished Run {run_count}/{total_runs}")
            except Exception as e:
                print(f"[X] Run {run_count} failed: {e}")
                
    print("\n" + "=" * 60)
    print("🎉 GRID SEARCH TUNING COMPLETE!")
    print("  Go to your DAGsHub Experiments tab to compare results!")
    print("=" * 60)

if __name__ == "__main__":
    run_grid_search()

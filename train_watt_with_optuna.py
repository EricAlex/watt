# train_watt_with_optuna.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import time
import os

# --- For reproducibility and to make this script self-contained, ---
# --- we include the necessary classes from the previous blocks. ---
# In a real project, these would be in separate files and imported.

from watt_model import WorldAwareTrajectoryTransformer # Assuming watt_model.py is in the same directory
from train_watt import SimulatedNuPlanDataset, WATT_Loss, move_to_device # Assuming train_watt.py is in the same directory

# --- 1. Validation Function (New Addition) ---
def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Use the robust move_to_device helper function
            batch = move_to_device(batch, device)
            images = batch['images']
            ground_truth = {k: v for k, v in batch.items() if k != 'images'}

            model_outputs = model(images)
            loss_dict = criterion(model_outputs, ground_truth)
            total_loss += loss_dict['total_loss'].item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- 2. The Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    """
    This function is called by Optuna for each trial.
    It builds, trains, and evaluates a model with a given set of hyperparameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- A. Suggest Hyperparameters ---
    # We define the search space for Optuna here.
    print(f"\n--- Starting Trial {trial.number} ---")
    
    # Architectural Hyperparameters
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 6)
    
    # Training Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    
    # Loss Weights
    bev_weight = trial.suggest_float("bev_weight", 0.2, 1.0)
    
    # Training Configuration
    BATCH_SIZE = 1 # Usually fixed based on GPU memory
    NUM_EPOCHS = 10 # Train for more epochs to get a reliable performance measure

    # --- B. Setup Model, Data, and Optimizer with Suggested Hyperparams ---
    
    # Model
    model_params = {
        'd_model': d_model, 'nhead': nhead,
        'num_encoder_layers': num_encoder_layers, 'num_decoder_layers': num_decoder_layers,
        'bev_height': 50, 'bev_width': 50, 'num_bev_classes': 3 # Smaller BEV for faster trials
    }
    model = WorldAwareTrajectoryTransformer(**model_params).to(device)
    
    # Data (use smaller datasets for faster trials)
    train_dataset = SimulatedNuPlanDataset(num_samples=400, bev_height=50, bev_width=50)
    val_dataset = SimulatedNuPlanDataset(num_samples=100, bev_height=50, bev_width=50)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Loss and Optimizer
    criterion = WATT_Loss(bev_weight=bev_weight)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

    # --- C. Run the Training and Validation Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # We reuse the train_one_epoch function from the previous block
        # For simplicity in this self-contained script, we'll inline a minimal version
        model.train()
        for batch in train_loader:
            batch = move_to_device(batch, device)
            images = batch['images']
            gt = {k: v for k, v in batch.items() if k != 'images'}
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, gt)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

        # Validation step
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Trial {trial.number}, Epoch {epoch}: Val Loss = {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # --- D. Report to Pruner and Return Final Metric ---
        # Report the intermediate value to the pruner.
        trial.report(val_loss, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number} finished. Best Val Loss: {best_val_loss:.5f}")
    return best_val_loss # Optuna will try to minimize this value

# --- 3. Main Execution Block to Run the Study ---
if __name__ == '__main__':
    # Best Choice: Use a storage backend to save results and allow resuming studies.
    # SQLite is a great, simple choice.
    storage_name = "sqlite:///watt_optuna.db"
    study_name = "watt-hyperparameter-search"

    print(f"Starting Optuna study. Results will be saved to {storage_name}")
    
    # Best Choice: Use a pruner to stop unpromising trials early.
    # MedianPruner stops trials that are performing worse than the median of other trials.
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, # Allows you to stop and resume the study
        direction="minimize", # We want to minimize validation loss
        pruner=pruner
    )

    # Start the optimization process. Optuna will run N trials.
    # You can parallelize this across multiple machines/GPUs if they share the storage.
    study.optimize(objective, n_trials=50)

    # --- Print Study Results ---
    print("\n--- Optuna Study Finished ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (min val_loss): {trial.value:.5f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also visualize the results if you install the optional dependencies
    # `pip install plotly kaleido`
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.write_image("optuna_history.png")
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.write_image("optuna_importances.png")
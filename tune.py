import os
import subprocess
import logging
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def objective(trial):
    # Suggest hyperparameters with defined ranges.
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 2e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16 ,32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.6)
    patience = trial.suggest_int("patience", 4, 6)
    epochs = 30  # Fixed number of epochs

    data_dir = "dataset"
    trial_name = f"lr{learning_rate:.0e}_bs{batch_size}_dr{dropout_rate:.2f}_pt{patience}"
    output_dir = os.path.join("hyperparameter_outputs", trial_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build the command to run the training script
    cmd = [
        "python", "classifier_training.py",
        "--data_dir", data_dir,
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--dropout_rate", str(dropout_rate),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--output_dir", output_dir
    ]

    logging.info("Starting trial: %s", trial_name)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Trial %s failed with error: %s", trial_name, e)
        # Prune the trial if the training script fails
        raise optuna.TrialPruned()

    # Read the performance metric from the output directory
    metric_file = os.path.join(output_dir, "avg_accuracy.txt")
    try:
        with open(metric_file, "r") as f:
            metric = float(f.read().strip())
    except Exception as e:
        logging.error("Trial %s: could not read metric file: %s", trial_name, e)
        # Prune the trial if no valid metric is found
        raise optuna.TrialPruned()

    logging.info("Trial %s finished with metric: %f", trial_name, metric)
    # Assuming that higher accuracy is better.
    return metric

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    # You can adjust n_trials based on your computational budget.
    study.optimize(objective, n_trials=15)

    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info("  Value: %f", best_trial.value)
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info("    %s: %s", key, value)

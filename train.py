import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import EarlyStopping
import warnings
import json

from gpu_limiter import GPULimiter
from process import preprocess
from model import VAE
from constants import (
    GPU_IDS,
    MAX_GPU_MEMORY_ALLOCATION,
    INTERMEDIATE_DIMS,
    LATENT_DIM,
    ORIGINAL_DIM,
    KERNEL_INITIALIZER,
    BIAS_INITIALIZER,
    BATCH_SIZES,
    LEARNIN_RATES,
    VALIDATION_SPLIT,
    EPOCHS,
    ACTIVATION,
    ACTIVATION_ETOT_DIV_ETRUTH,
    GLOBAL_CHECKPOINT_DIR,
    PATIENCE,
    BETAS_MAX,
    BETAS_MIN,
    BETA_PERIOD,
)

# imports for pruning
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import (
    prune_low_magnitude,
    strip_pruning,
    UpdatePruningStep,
    PruningSummaries,
    )
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import ConstantSparsity


class BetaCosineCallback(keras.callbacks.Callback):
    """Updates vae.beta each epoch following a cosine oscillation."""
    def __init__(self, beta_var, beta_min, beta_max, period):
        super().__init__()
        self.beta_var = beta_var
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.period   = period

    def on_epoch_begin(self, epoch, logs=None):
        import math
        t    = epoch % self.period
        beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (
            1 - math.cos(math.pi * t / (self.period / 2))
        )
        self.beta_var.assign(beta)


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--file-name", type=str, default="dataset/dataset_1_photons_1.hdf5"
    )
    argument_parser.add_argument("--out-dir", type=str, required=True)
    argument_parser.add_argument("--sparsity",type=float,required=True)
    argument_parser.add_argument("--bits",type=int, required=True)
    argument_parser.add_argument("--gpu-ids", type=str, default=GPU_IDS)
    argument_parser.add_argument(
        "--max-gpu-memory-allocation", type=int, default=MAX_GPU_MEMORY_ALLOCATION
    )
    args = argument_parser.parse_args()
    return args


def main():
    """
    # Filter out RuntimeWarnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Filter out TensorFlow warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    """

    # Parse arguments.
    args = parse_args()
    file_name = args.file_name
    out_dir = args.out_dir
    sparsity = args.sparsity
    bits = args.bits
    gpu_ids = args.gpu_ids
    max_gpu_memory_allocation = args.max_gpu_memory_allocation

    # build the full checkpoint path and create it (and parents) if it doesn't already exist
    checkpoint_root = os.path.join(GLOBAL_CHECKPOINT_DIR, out_dir)
    os.makedirs(checkpoint_root, exist_ok=True)

    # Set GPU memory limits.
    GPULimiter(_gpu_ids=gpu_ids, _max_gpu_memory_allocation=max_gpu_memory_allocation)()

    # Data loading/preprocessing
    # The preprocess function reads the data and performs preprocessing and encoding for the values of energy,
    energies_train, cond_e_train = preprocess(file_name)

    # Test with different batch sizes and learning rates
    for test_version in range(len(LEARNIN_RATES)):
        print(f" ....... Currently test_version {test_version} ....... ")

        # Instantiate the VAE model
        vae = VAE(
            original_dim=ORIGINAL_DIM,
            intermediate_dim1=INTERMEDIATE_DIMS[0],
            intermediate_dim2=INTERMEDIATE_DIMS[1],
            intermediate_dim3=INTERMEDIATE_DIMS[2],
            intermediate_dim4=INTERMEDIATE_DIMS[3],
            latent_dim=LATENT_DIM,
            kernel_initializer=KERNEL_INITIALIZER,
            bias_initializer=BIAS_INITIALIZER,
            activation=ACTIVATION,
            activ_frac_etot_etruth=ACTIVATION_ETOT_DIV_ETRUTH,
            optimizer=tf.optimizers.Adam(LEARNIN_RATES[test_version]),
            w_reco=ORIGINAL_DIM,
            sparsity=sparsity,
            bits=bits
        )

        # Load previous best model
        if test_version > 1:
            prev_prefix = f"VAE-V{test_version-1}-"
            files = [f for f in os.listdir(checkpoint_root)
                    if f.startswith(prev_prefix) and f.endswith("-weights.h5")]
            if files:
                files.sort(key=lambda x: os.path.getctime(os.path.join(checkpoint_root, x)))
                last_added_file = files[-1]
                vae.vae.load_weights(os.path.join(checkpoint_root, last_added_file))
                print("weights loaded",os.path.join(checkpoint_root, last_added_file))


        # The model checkpoint callback

        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_root, f"VAE-V{test_version}-" + "{epoch:02d}-weights.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,   
            mode="auto",
            save_freq="epoch",        
)


        # The early stopping callback
        callback_early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            verbose=1,
            mode="auto",
        )

        
        callback_beta = BetaCosineCallback(
            beta_var=vae.beta,
            beta_min=BETAS_MIN[test_version],
            beta_max=BETAS_MAX[test_version],
            period=BETA_PERIOD,
        )

        callbacks = [
            callback_checkpoint,
            callback_early_stopping,
            callback_beta,
            UpdatePruningStep(),
            PruningSummaries(log_dir=os.path.join(checkpoint_root, 'pruning')),
        ]

        # Train the VAE model
        noise = np.random.normal(0, 1, size=(energies_train.shape[0], LATENT_DIM))
        output = vae.vae.fit(
            [energies_train, cond_e_train, noise],
            [energies_train],
            shuffle=True,
            verbose=1,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            batch_size=BATCH_SIZES[test_version],
            callbacks=callbacks,
        ) #[callback_checkpoint, callback_early_stopping],

        print(output.history)
        with open(f"{checkpoint_root}/hist_{test_version}.json", "w", encoding="utf-8") as f:
            json.dump(output.history, f, indent=2)
        


if __name__ == "__main__":
    exit(main())

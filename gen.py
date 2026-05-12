#generation speed test

import os, argparse, time
# Pin to a single GPU for consistent timing (must be set before importing TF)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter logs

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning

from model import VAE
from process import load_incident_energies, energy_to_onehot
from constants import (
    ORIGINAL_DIM, INTERMEDIATE_DIMS, LATENT_DIM,
    KERNEL_INITIALIZER, BIAS_INITIALIZER, ACTIVATION,
    ACTIVATION_ETOT_DIV_ETRUTH, GLOBAL_CHECKPOINT_DIR,
)

# python gen.py --check-dir Qkeras_P85Q6 --version 7 --epoch 9 --sparsity 0.85 --bits 6

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-dir", type=str, required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--bits", type=int, required=True)  
    ap.add_argument("--data", type=str, default="dataset/dataset_1_photons_1.hdf5")
    return ap.parse_args()

def main():
    args = parse_args()

    # Instantiate VAE and load weights
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
        optimizer=tf.optimizers.Adam(),
        w_reco=ORIGINAL_DIM,
        sparsity=args.sparsity,
        bits=args.bits,
    )

    checkpoint_root = os.path.join(GLOBAL_CHECKPOINT_DIR, args.check_dir)
    os.makedirs(checkpoint_root, exist_ok=True)
   
    ckpt_path = os.path.join(checkpoint_root, f"VAE-V{args.version}-{args.epoch:02d}-weights.h5")
    vae.vae.load_weights(ckpt_path)

    # Strip pruning wrappers and use the stripped decoder
    vae.vae = strip_pruning(vae.vae)
    decoder = strip_pruning(vae.decoder)

    # ----- generation inputs -----
    incident_energies, max_energy = load_incident_energies(args.data)
    
    # Device label
    dev_label = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    N = 100000 #number of samples

    def bench(batch_sizes=(1, 100, 10000), repeats=11): #repeat and drop first time (warmup)
        for bs in batch_sizes:
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                z_n_gauss = np.random.normal(loc=0, scale=1, size=(len(incident_energies), LATENT_DIM))
                scalar_cond = (np.log2(incident_energies) / np.log2(max_energy)).reshape(-1, 1)
                onehot_cond = energy_to_onehot(incident_energies)
                z = np.column_stack([z_n_gauss, scalar_cond, onehot_cond]).astype(np.float32)
                zN = z[:N].astype(np.float32)
                _  = decoder.predict(zN, batch_size=bs, verbose=0)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            t = np.array(times[1:], dtype=np.float64)

            avg_s = t.mean()
            std_s = t.std(ddof=1) if len(t) > 1 else 0.0  # sample std dev

            ms_per = avg_s / N * 1e3
            std_ms_per = std_s / N * 1e3

            print(
                f"{dev_label}  bs={bs:>6}: {ms_per:.5f} ± {std_ms_per:.5f} ms/shower  "
                f"(avg {avg_s:.3f} s, std {std_s:.3f} s over {len(t)} runs, N={N})"
            )

    bench()


if __name__ == "__main__":
    main()

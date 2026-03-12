import argparse
import h5py
import tensorflow as tf
import numpy as np
import os
from model import VAE
from process import postprocess, load_incident_energies
from constants import (
    ORIGINAL_DIM,
    INTERMEDIATE_DIMS,
    LATENT_DIM,
    KERNEL_INITIALIZER,
    BIAS_INITIALIZER,
    ACTIVATION,
    ACTIVATION_ETOT_DIV_ETRUTH,
    GLOBAL_CHECKPOINT_DIR,
)

from tensorflow_model_optimization.sparsity.keras import strip_pruning


#  python HLS_model.py --check-dir P75 --version 1 --epoch 65 --sparsity 0.75
'''
exports 1. decoder model, 
        2. golden in/out
        3. VAE_pions (final product to compare with geant 4)
to decdoer_models directory
'''

def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--check-dir", type=str, required=True)
    argument_parser.add_argument("--version", type=int, required=True)
    argument_parser.add_argument("--epoch", type=int, required=True)
    argument_parser.add_argument("--sparsity",type=float,required=True)
    argument_parser.add_argument("--bits",type=int, required=True)
    argument_parser.add_argument("--data", type=str, default="dataset/dataset_1_photons_1.hdf5")
    return argument_parser.parse_args()


def main():
    #parse args
    args = parse_args()
    check_dir = args.check_dir
    version = args.version
    epoch = args.epoch
    sparsity = args.sparsity
    bits = args.bits
    file_name = args.data

    #Instantiate VAE to access the decoder
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
        sparsity=sparsity,
        bits=bits
    )

    #make output directory for all 3 outputs under decoder_models/check_dir (with out dir same name as checkpoint dir)
    output_root = os.path.join("decoder_models", check_dir)
    os.makedirs(output_root, exist_ok=True)

    # Load trained weights
    checkpoint_root = os.path.join(GLOBAL_CHECKPOINT_DIR, check_dir)
    os.makedirs(checkpoint_root, exist_ok=True)
    

    ckpt_path = os.path.join(checkpoint_root, f"VAE-V{version}-{epoch:02d}-weights.h5")
    vae.vae.load_weights(ckpt_path)
    
    #for prunning
    vae.vae = strip_pruning(vae.vae)
    
    #save decoder (architecture + weights) in HDF5 format
    decoder = strip_pruning(vae.decoder)
    decoder_filename = f"{output_root}/decoder-{check_dir}-V{version}-{epoch}.h5" #change name for new models
    decoder.save(decoder_filename)
    

    
    #-----generation------

    # Load the incident energies, generate latent space, and create variables
    incident_energies, max_energy = load_incident_energies(file_name)
    z_n_gauss = np.random.normal(loc=0, scale=1, size=(len(incident_energies), LATENT_DIM))

    z = np.column_stack((z_n_gauss, np.log2(incident_energies) / np.log2(max_energy))) #now log scaled and normalized
    predicted_energies = vae.decoder.predict(z)

    
    #save 'z' and 'incident_energies' as the 'golden input/output' for the decoder as a single .h5
    golden_filename = f"{output_root}/golden-{check_dir}-V{version}-{epoch}.h5"
    with h5py.File(golden_filename, "w") as f:
        f.create_dataset("z",        data=z)
        f.create_dataset("energies", data=predicted_energies)
    


    #---- rescaling for after hls4ml (and here to properly run eval) ---- needs process.py to run and GPU access

    predicted_energies = postprocess(predicted_energies, incident_energies) #Post process uses straight inc_energies so no need to have log scale
    print(f"Generation completed of {len(predicted_energies)} showers.")

    # Save the hdf5 file (same format as the Geant4 dataset)
    dataset_file = h5py.File(f"{output_root}/VAE_photons-{check_dir}-V{version}-{epoch}.hdf5", "w")
    dataset_file.create_dataset(
        "incident_energies",
        data=incident_energies.reshape(len(incident_energies), -1),
        compression="gzip",
    )
    dataset_file.create_dataset(
        "showers",
        data=predicted_energies.reshape(len(predicted_energies), -1),
        compression="gzip",
    )
    dataset_file.close()
    


if __name__ == "__main__":
    main()


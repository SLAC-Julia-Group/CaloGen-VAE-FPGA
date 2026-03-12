This repository contains the scripts for training and exporting a compressed generative VAE model for the CaloChallenge [Dataset (photons_1)](https://zenodo.org/records/8099322#:~:text=Download-,dataset_1_photons_1.hdf5,-md5%3A6a5f52722064a1bcd8a0bc002f16515d). The code is heavily based on the work of [ATLASDNNCaloSim](https://github.com/DalilaSalamani/ATLASDNNCaloSim). The code for synthesizing the model to HLS code with hls4ml is not in this repository.

- `setup.py`: creates needed directories.
- `process.py`: defines the data loading, preprocessing, and postprocessing functions.
- `gpu_limiter.py`: defines a logic responsible for GPU memory management.
- `constants.py`: defines the set of common variables.
- `model.py`: defines the VAE model class and a handler to construct the model.
- `train.py`: performs model training.
- `HLS_model.py`: performs generation and saves a model file for hls4ml.
- `gen.py`: gives generation timing results.
- `evaluation/evaluate.py`: evaluates the performance of the trained model

`decoder_models/QKeras_P85Q6/decoder...` contains the pre-trained decoder of the VAE for FPGA synthesis.


## Getting Started 

Create environment:
```
conda env create -f environment.yml
```

`setup.py` script creates necessary folders used to save model checkpoints, generate showers and validation plots.

```
python setup.py
``` 

## Training

To run the training with specified sparsity (pruning) and number of bits allocated to QDense layers (quantized_bits(bits,0,1) and quantized_relu(bits,0) ):
```
python train.py --out-dir QKeras_P85Q6 --sparsity 0.85 --bits 6
```


## Generation/Model

Generates shower samples as an HDF5 file used for evaluation, "golden" decoder inputs (latent vector z) and outputs (before postprocess rescaling) as an h5, and the decoder model itself as an h5 file for hls4ml. Exports all 3 files to decoder_model/...:
```
python HLS_model.py --check-dir QKeras_P85Q6 --sparsity 0.85 --bits 6 --version 7 --epoch 11
``` 

## Generation Speed Benchmark

Runs multiple generations without saving outputs and returns average generation speed for batch sizes of 1, 100, 10000:
```
python gen.py --check-dir QKeras_P85Q6 --version 7 --epoch 11 --sparsity 0.85 --bits 6
```

## Evaluation

The evaluation is based on the [CaloChallenge code](https://github.com/CaloChallenge/homepage). Go to the evaluation folder and run:
```
python evaluate.py -i ../decoder_models/QKeras_P85Q6/VAE_photons-QKeras_P85Q6-V7-11.hdf5 -r ../dataset/dataset_1_photons_2.hdf5 -m all -d 1-photons --output_dir eval_plots/QKeras_P85Q6/V7-11
```

import tensorflow as tf

"""
Directories.
"""
# Directory to load the full simulation dataset.
INIT_DIR = "./dataset"
# Directory to save VAE checkpoints
GLOBAL_CHECKPOINT_DIR = "./checkpoints"
# Directory to save decoder model to use on hls4ml
CONV_DIR = "./decoder_models"
# Directory to save validation plots.
VALID_DIR = "./validation"
# Directory to save VAE generated showers.
GEN_DIR = "./generations"

"""
GPU resources
"""
# GPU identifiers separated by comma, no spaces.
GPU_IDS = "0"
# Maximum allowed memory on one of the GPUs (in GB)
MAX_GPU_MEMORY_ALLOCATION = 32

"""
Experiment constants (ds1 – γ / photons)
Geometry from the FastCalo Challenge table:
L0=8, L1=16x10=160, L2=19x10=190, L3=5, L12=5; 
"""
# Number of layers considered
N_LAYERS = 5

# Per-layer voxel counts
N_VOXELS_L0 = 8
N_VOXELS_L1 = 16 * 10  # 160
N_VOXELS_L2 = 19 * 10  # 190
N_VOXELS_L3 = 5
N_VOXELS_L12 = 5

# Absent for photons; 
N_VOXELS_L13 = 0
N_VOXELS_L14 = 0

# List of voxel counts in the order they appear in the flat shower vector
# (L0, L1, L2, L3, L12). Helpful for generic slicing.
VOXEL_COUNTS = [N_VOXELS_L0, N_VOXELS_L1, N_VOXELS_L2, N_VOXELS_L3, N_VOXELS_L12]

# Total number of voxels across active layers
N_VOXELS = sum(VOXEL_COUNTS)

# Input vector structure:
# [all voxel ratios | (Etot/Einc)/scale | per-layer energy fractions (N_LAYERS)]
ORIGINAL_DIM = N_VOXELS + 1 + N_LAYERS

#manual scaling: zeta
ETOT_EINC_SCALE = 1.6

# One-hot encoding of the 15 discrete incident energies (2^8 ... 2^22 MeV)
N_ENERGY_BINS = 15
E_INC_LOG2_MIN = 8     # 256 MeV = 2^8
E_INC_LOG2_MAX = 22    # 4 TeV = 2^22
COND_DIM = 1 + N_ENERGY_BINS   # scalar + one-hot = 16


"""
Model & training parameters
"""
INTERMEDIATE_DIMS = [150, 120, 90, 60]  # [1500, 1000, 500, 100] original
LATENT_DIM = 30    # 50 original
KERNEL_INITIALIZER = "RandomNormal"
BIAS_INITIALIZER = "Zeros"
ACTIVATION = 'leaky_relu'
ACTIVATION_ETOT_DIV_ETRUTH = 'sigmoid'
BATCH_SIZES = [1000, 500, 1000, 250, 1000, 500, 250, 1000] # large option (faster training, worse results) BATCH_SIZES = [1000, 500, 250, 1000, 1000, 1000, 500, 5000]
LEARNIN_RATES = [0.01, 0.001, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001] 
VALIDATION_SPLIT = 0.15
EPOCHS = 100000
# Number of epochs with no improvement after which training will be stopped
#PATIENCE = 10

# Cosine oscillating beta schedule, one entry per stage 
# BETAS_MAX : peak KL weight reached at each half-cycle
# BETAS_MIN : min KL weight (bottom of each oscillation)
# BETA_PERIOD: epochs for one full cosine cycle (min, max, min)
PATIENCE = 30
BETAS_MIN = [0.0, 0.0, 0.0, 0.1, 0.25, 0.4, 0.6, 0.8]   # rising troughs
BETAS_MAX = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
BETA_PERIOD = 20

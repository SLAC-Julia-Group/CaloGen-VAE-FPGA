#model with only prunning
#BASE - Keras
#to have VAE-large change INTERMEDIATE_DIMS = [1500, 1000, 500, 100] and LATENT_DIM = 50
##to have VAE-small change INTERMEDIATE_DIMS = [150, 120, 90, 60] and LATENT_DIM = 30

import numpy as np
import tensorflow as tf
from keras import metrics
from keras import backend as K
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from keras.layers import (
    Input,
    Dense,
    Lambda,
    Layer,
    Multiply,
    Add,
    Concatenate,
)

from constants import (
    N_VOXELS_L0,
    N_VOXELS_L1,
    N_VOXELS_L2,
    N_VOXELS_L3,
    N_VOXELS_L12,
    N_LAYERS
)

# pruning
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import ConstantSparsity


class VAE:
    def __init__(self, **kwargs):
        self.original_dim = kwargs.get("original_dim")
        self.intermediate_dim1 = kwargs.get("intermediate_dim1")
        self.intermediate_dim2 = kwargs.get("intermediate_dim2")
        self.intermediate_dim3 = kwargs.get("intermediate_dim3")
        self.intermediate_dim4 = kwargs.get("intermediate_dim4")
        self.latent_dim = kwargs.get("latent_dim")
        self.kernel_initializer = kwargs.get("kernel_initializer")
        self.bias_initializer = kwargs.get("bias_initializer")
        self.activation = kwargs.get("activation")
        self.activ_frac_etot_etruth = kwargs.get("activ_frac_etot_etruth")
        self.optimizer = kwargs.get("optimizer")
        self.w_reco = kwargs.get("w_reco")
        self.sparsity = kwargs.get("sparsity", 0.0)
        self.bits = kwargs.get("bits",8) #to be compatible with training that is designed for quantized model

        class KLDivergenceLayer(Layer):
            """Identity transform layer that adds KL divergence to the final model loss."""
            def __init__(self, *args, **kwargs):
                self.is_placeholder = True
                super(KLDivergenceLayer, self).__init__(*args, **kwargs)

            def call(self, inputs):
                mu, log_var = inputs
                kl_batch = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
                self.add_loss(K.mean(kl_batch), inputs=inputs)
                return inputs

        # -------------------
        # Encoder
        # -------------------
        x = Input(shape=(self.original_dim,))
        e_cond = Input(shape=(1,))
        merged_input = Concatenate(axis=-1)([x, e_cond])

        h1 = Dense(self.intermediate_dim1, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(merged_input)
        h1 = BatchNormalization()(h1)
        h2 = Dense(self.intermediate_dim2, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h1)
        h2 = BatchNormalization()(h2)
        h3 = Dense(self.intermediate_dim3, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h2)
        h3 = BatchNormalization()(h3)
        h4 = Dense(self.intermediate_dim4, activation=self.activation,
                   kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(h3)
        h = BatchNormalization()(h4)

        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

        z_sigma = Lambda(lambda t: K.exp(0.5 * t))(z_log_var)
        # feed noise from train.py
        eps = Input(shape=(self.latent_dim,))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        z_cond = Concatenate(axis=-1)([z, e_cond])

        # public encoder
        self.encoder = Model(inputs=[x, e_cond, eps], outputs=z_cond)

        # -------------------
        # Decoder / Generator
        # -------------------
        deco_l4 = Dense(self.intermediate_dim4, input_dim=(self.latent_dim + 1),
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l4_bn = BatchNormalization()
        deco_l3 = Dense(self.intermediate_dim3, input_dim=self.intermediate_dim4,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l3_bn = BatchNormalization()
        deco_l2 = Dense(self.intermediate_dim2, input_dim=self.intermediate_dim3,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l2_bn = BatchNormalization()
        deco_l1 = Dense(self.intermediate_dim1, input_dim=self.intermediate_dim2,
                        activation=self.activation, kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)
        deco_l1_bn = BatchNormalization()
        deco_output = Dense(self.original_dim, activation=None) #None from self.activation

        # training path: z_cond -> MLP -> per-branch heads -> concat
        x_reco = deco_output(
            deco_l1_bn(
                deco_l1(
                    deco_l2_bn(
                        deco_l2(deco_l3_bn(deco_l3(deco_l4_bn(deco_l4(z_cond)))))
                    )
                )
            )
        )

        # voxel heads (photons: L0, L1, L2, L3, L12)
        nodes_l0  = Dense(N_VOXELS_L0,  activation="softmax")(x_reco)
        nodes_l1  = Dense(N_VOXELS_L1,  activation="softmax")(x_reco)
        nodes_l2  = Dense(N_VOXELS_L2,  activation="softmax")(x_reco)
        nodes_l3  = Dense(N_VOXELS_L3,  activation="softmax")(x_reco)
        nodes_l12 = Dense(N_VOXELS_L12, activation="softmax")(x_reco)
        # Etot/Einc scalar and per-layer fractions
        node_etot_etruth = Dense(1,        activation=self.activ_frac_etot_etruth)(x_reco)
        node_layers_frac = Dense(N_LAYERS, activation="softmax")(x_reco)

        x_reco_final = Concatenate(axis=-1)([
            nodes_l0, nodes_l1, nodes_l2, nodes_l3, nodes_l12,
            node_etot_etruth, node_layers_frac
        ])

        # inference decoder: input is (z, e_cond) concat from encoder
        z_deco_input = Input(shape=(self.latent_dim + 1,))
        x_reco_deco = deco_output(
            deco_l1_bn(
                deco_l1(
                    deco_l2_bn(
                        deco_l2(deco_l3_bn(deco_l3(deco_l4_bn(deco_l4(z_deco_input)))))
                    )
                )
            )
        )

        nodes_l0_reco  = Dense(N_VOXELS_L0,  activation="softmax")(x_reco_deco)
        nodes_l1_reco  = Dense(N_VOXELS_L1,  activation="softmax")(x_reco_deco)
        nodes_l2_reco  = Dense(N_VOXELS_L2,  activation="softmax")(x_reco_deco)
        nodes_l3_reco  = Dense(N_VOXELS_L3,  activation="softmax")(x_reco_deco)
        nodes_l12_reco = Dense(N_VOXELS_L12, activation="softmax")(x_reco_deco)

        node_etot_etruth_reco = Dense(1,        activation=self.activ_frac_etot_etruth)(x_reco_deco)
        node_layers_frac_reco = Dense(N_LAYERS, activation="softmax")(x_reco_deco)

        # pairwise merges (photons: L0, L1, L2, L3, L12 + Etot/Einc + layer fracs)
        c01      = Concatenate(axis=-1)([nodes_l0_reco, nodes_l1_reco])
        c23      = Concatenate(axis=-1)([nodes_l2_reco, nodes_l3_reco])
        c45      = Concatenate(axis=-1)([nodes_l12_reco, node_etot_etruth_reco])
        c0123    = Concatenate(axis=-1)([c01, c23])
        c012345  = Concatenate(axis=-1)([c0123, c45])
        x_reco_final_reco = Concatenate(axis=-1)([c012345, node_layers_frac_reco])


        self.decoder = Model(inputs=[z_deco_input], outputs=[x_reco_final_reco])

        # prune decoder only if requested
        if self.sparsity > 0:
            pruning_params = {"pruning_schedule": ConstantSparsity(self.sparsity, begin_step=2000, frequency=100)}
            self.decoder = prune_low_magnitude(self.decoder, **pruning_params)

        # -------------------
        # VAE end-to-end
        # -------------------
        def vae_loss(g4_event, vae_event):
            return self.w_reco * K.sum(metrics.binary_crossentropy(g4_event, vae_event))

        self.vae = Model(inputs=[x, e_cond, eps],
                         outputs=[self.decoder(self.encoder([x, e_cond, eps]))])
        self.vae.compile(optimizer=self.optimizer, loss=vae_loss)

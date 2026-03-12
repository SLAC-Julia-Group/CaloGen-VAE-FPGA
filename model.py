#qauntized decoder and sparsity schedule

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
    Activation
)

from constants import (
    N_VOXELS_L0,
    N_VOXELS_L1,
    N_VOXELS_L2,
    N_VOXELS_L3,
    N_VOXELS_L12,
    N_LAYERS,
)

# pruning
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import ConstantSparsity
#quantization
from qkeras import QDense, quantized_bits, QActivation, QBatchNormalization
from tensorflow.keras.layers import LeakyReLU



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
        self.bits = kwargs.get("bits",8) # adjusts kernel and activation bits

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

        #  encoder
        self.encoder = Model(inputs=[x, e_cond, eps], outputs=z_cond)


        # -------------------
        # Decoder / Generator 
        # -------------------
  
        # Quantization parameters for QDense
        wk_bits = self.bits
        wi_bits = 1
        bk_bits = self.bits + 2
        bi_bits = 2 

        # QDense Trunk quantization
        w_q   = quantized_bits(bits=wk_bits, integer=wi_bits, symmetric=1, alpha=1)
        b_q   = quantized_bits(bits=bk_bits, integer=bi_bits, symmetric=1, alpha=1)
        # QDense Head quantization
        w_q_heads  = quantized_bits(bits=wk_bits,   integer=wi_bits, symmetric=1, alpha=1)
        b_q_heads  = quantized_bits(bits=bk_bits,   integer=bi_bits, symmetric=1, alpha=1)
        w_q_energy = quantized_bits(bits=wk_bits+2, integer=wi_bits+1, symmetric=1, alpha=1)
        b_q_energy = quantized_bits(bits=bk_bits+2, integer=bi_bits, symmetric=1, alpha=1)

        #QBatchNorm quantization
        beta_q =  quantized_bits(bits=8, integer=3, symmetric=1, alpha=1)
        gamma_q = quantized_bits(bits=8, integer=2, symmetric=1, alpha=1)
        mean_q =  quantized_bits(bits=10, integer=4, symmetric=1, alpha=1)
        var_q =   quantized_bits(bits=12, integer=6, symmetric=1, alpha=1)

        leaky_slope = 0.25
        def trunk_block(x, units, dense_name, bn_name, leaky_name):
            x = QDense(
                name = dense_name,
                units=units,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_quantizer=w_q,
                bias_quantizer=b_q,
                activation=None)(x)
            x = QBatchNormalization(name=bn_name,
                beta_quantizer=beta_q,
                gamma_quantizer=gamma_q,
                mean_quantizer=mean_q,
                variance_quantizer=var_q)(x)
            x = LeakyReLU(alpha=leaky_slope, name=leaky_name)(x)
            return x


        # Input
        z_deco_input = Input(shape=(self.latent_dim + 1,), name="z_deco_input")

        # Trunk
        h4 = trunk_block(
            x=z_deco_input,
            units=self.intermediate_dim4,
            dense_name="q_dense_0", bn_name="batch_normalization_D0", leaky_name="leaky_0",
        )
        h3 = trunk_block(
            x=h4,
            units=self.intermediate_dim3,
            dense_name="q_dense_1", bn_name="batch_normalization_D1", leaky_name="leaky_1",
        )
        h2 = trunk_block(
            x=h3,
            units=self.intermediate_dim2,
            dense_name="q_dense_2", bn_name="batch_normalization_D2", leaky_name="leaky_2",
        )
        h1 = trunk_block(
            x=h2,
            units=self.intermediate_dim1,
            dense_name="q_dense_3", bn_name="batch_normalization_D3", leaky_name="leaky_3",
        )

        # Final trunk stays linear 
        deco_output = QDense(
            units=self.original_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q,
            bias_quantizer=b_q,
            activation=None,
            name="q_dense_4",
        )

        x_reco_deco = deco_output(h1)

        # Heads/Branches
        nodes_l0_reco  = Activation('softmax',name='activation_l0')(QDense(N_VOXELS_L0,  activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_heads, bias_quantizer=b_q_heads, name = 'q_dense_l0')(x_reco_deco))
        nodes_l1_reco  = Activation('softmax',name='activation_l1')(QDense(N_VOXELS_L1,  activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_heads, bias_quantizer=b_q_heads, name = 'q_dense_l1')(x_reco_deco))
        nodes_l2_reco  = Activation('softmax',name='activation_l2')(QDense(N_VOXELS_L2,  activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_heads, bias_quantizer=b_q_heads, name = 'q_dense_l2')(x_reco_deco))
        nodes_l3_reco  = Activation('softmax',name='activation_l3')(QDense(N_VOXELS_L3,  activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_heads, bias_quantizer=b_q_heads, name = 'q_dense_l3')(x_reco_deco))
        nodes_l12_reco = Activation('softmax',name='activation_l12')(QDense(N_VOXELS_L12, activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_heads, bias_quantizer=b_q_heads, name = 'q_dense_l12')(x_reco_deco))

        node_etot_etruth_reco = Dense(1, activation=self.activ_frac_etot_etruth,name = 'dense_etot')(x_reco_deco)

        node_layers_frac_reco = Activation('softmax',name = 'activation_LFR')(QDense(N_LAYERS, activation='linear',
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_quantizer=w_q_energy, bias_quantizer=b_q_energy, name = 'q_dense_LFR')(x_reco_deco))

        # Concats
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
        # total VAE 
        # -------------------
        def vae_loss(g4_event, vae_event):
            return self.w_reco * K.sum(metrics.binary_crossentropy(g4_event, vae_event))

        self.vae = Model(inputs=[x, e_cond, eps],
                         outputs=[self.decoder(self.encoder([x, e_cond, eps]))])
        self.vae.compile(optimizer=self.optimizer, loss=vae_loss, metrics=["mse","mae","binary_crossentropy"])

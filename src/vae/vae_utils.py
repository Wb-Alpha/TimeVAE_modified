from typing import Union, List, Optional
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import warnings
from vae.vae_base import Sampling

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

import tensorflow as tf

from vae.vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae.vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae.timevae import TimeVAE


def set_seeds(seed: int = 111) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    # Set the seed for TensorFlow
    tf.random.set_seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python built-in random module
    random.seed(seed)


def instantiate_vae_model(
    vae_type: str, sequence_length: int, feature_dim: int, batch_size: int, **kwargs
) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    """
    Instantiate a Variational Autoencoder (VAE) model based on the specified type.

    Args:
        vae_type (str): The type of VAE model to instantiate.
                        One of ('vae_dense', 'vae_conv', 'timeVAE').
        sequence_length (int): The sequence length.
        feature_dim (int): The feature dimension.
        batch_size (int): Batch size for training.

    Returns:
        Union[VAE_Dense, VAE_Conv, TimeVAE]: The instantiated VAE model.

    Raises:
        ValueError: If an unrecognized VAE type is provided.
    """
    set_seeds(seed=123)

    if vae_type == "vae_dense":
        vae = VAE_Dense(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "vae_conv":
        vae = VAE_Conv(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    elif vae_type == "timeVAE":
        vae = TimeVAE(
            seq_len=sequence_length,
            feat_dim=feature_dim,
            batch_size=batch_size,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unrecognized model type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def train_vae(vae, train_data, max_epochs, verbose=0):
    """
    Train a VAE model.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The VAE model to train.
        train_data (np.ndarray): The training data which must be of shape
                                 [num_samples, window_len, feature_dim].
        max_epochs (int, optional): The maximum number of epochs to train
                                    the model.
                                    Defaults to 100.
        verbose (int, optional): Verbose arg for keras model.fit()
    """
    vae.fit_on_data(train_data, max_epochs, verbose)


def save_vae_model(vae, dir_path: str) -> None:
    """
    Save the weights of a VAE model.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The VAE model to save.
        dir_path (str): The directory to save the model weights.
    """
    vae.save(dir_path)


def load_vae_model(vae_type: str, dir_path: str) -> Union[VAE_Dense, VAE_Conv, TimeVAE]:
    """
    Load a VAE model from the specified directory.

    Args:
        vae_type (str): The type of VAE model to load.
                        One of ('vae_dense', 'vae_conv', 'timeVAE').
        dir_path (str): The directory containing the model weights.

    Returns:
        Union[VAE_Dense, VAE_Conv, TimeVAE]: The loaded VAE model.
    """
    if vae_type == "vae_dense":
        vae = VAE_Dense.load(dir_path)
    elif vae_type == "vae_conv":
        vae = VAE_Conv.load(dir_path)
    elif vae_type == "timeVAE":
        vae = TimeVAE.load(dir_path)
    else:
        raise ValueError(
            f"Unrecognized model type [{vae_type}]. "
            "Please choose from vae_dense, vae_conv, timeVAE."
        )

    return vae


def get_posterior_samples(vae, data):
    vae.summary()
    """
    Get posterior samples from the VAE model.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The trained VAE model.
        data (np.ndarray): The data to generate posterior samples from.

    Returns:
        np.ndarray: The posterior samples.
    """
    # 最后
    # enc_conv_{i}
    # encoder_input = tf.keras.models.Model(inputs=vae.encoder.input, outputs=vae.encoder.get_layer('enc_conv_1').output)
    # enc_conv_1 = tf.keras.models.Model(inputs=vae.encoder.input, outputs=vae.encoder.get_layer('enc_conv_1').output) # 注意
    # enc_conv_2 = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('enc_conv_2').output) # 注意
    # enc_conv_3 = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('enc_conv_3').output) # 注意
    # enc_flatten = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('enc_flatten').output)
    # z_mean = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('z_mean').output) # 注意
    # level_para_1 = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('level_params1').output) # 注意
    # level_para_2 = tf.keras.models.Model(inputs=vae.input, outputs=vae.get_layer('level_params2').output) # 注意
    # # dec_dense = tf.keras.models.Model(inputs=self.decoder.input, outputs=self.decoder.get_layer('dec_dense').output)
    # enc_conv_1_output = enc_conv_1.predict(x=data, verbose=0) # 791,22,100
    # print(level_para_1_output)
    # dec_deconv_0
    # dec_deconv_1
    # dec_deconv_2
    # dec_deconv_3
    # dec_flatten
    # decoder_dense_final
    encoder_input_fm = get_layer_feature(vae.encoder,'encoder_input',data)
    enc_conv_1 = get_layer_feature(vae.encoder,'enc_conv_0',data)
    enc_conv_2 = get_layer_feature(vae.encoder,'enc_conv_1',data)
    enc_conv_3 = get_layer_feature(vae.encoder,'enc_conv_2',data)
    enc_flatten = get_layer_feature(vae.encoder,'enc_flatten',data)
    z_mean = get_layer_feature(vae.encoder,'z_mean',data)
    print("z_mean")
    print(z_mean)
    z_log_var = get_layer_feature(vae.encoder,'z_log_var',data)
    print("var")
    print(z_log_var)
    encoder_output = Sampling()([z_mean, z_log_var])
    print(encoder_output)
    level_para_1 = get_layer_feature(vae.decoder,'decoder_input',encoder_output)
    print("level_para_1")
    print(level_para_1)
    dec_dense = get_layer_feature(vae.decoder,'dec_dense',encoder_output)
    print("dense")
    print(dec_dense)
    dec_conv_1_output = get_layer_feature(vae.decoder,'residual_reshape',encoder_output)
    print("dec_conv_1_output")
    print(dec_conv_1_output)
    output = vae.predict(data, verbose=0) # output=data: 791, 85, 5

    return output,z_mean,z_log_var

def get_layer_feature(component, name, data):
    feature_map = tf.keras.models.Model(inputs=component.input, outputs=component.get_layer(name).output)
    enc_conv_1_output = feature_map.predict(x=data, verbose=0) # 791,22,100
    return enc_conv_1_output



def get_prior_samples(vae, num_samples: int):
    """
    Get prior samples from the VAE model.

    Args:
        vae (Union[VAE_Dense, VAE_Conv, TimeVAE]): The trained VAE model.
        num_samples (int): The number of samples to generate.

    Returns:
        np.ndarray: The prior samples.
    """
    return vae.get_prior_samples(num_samples=num_samples)

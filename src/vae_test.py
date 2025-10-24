import os, warnings
import numpy as np
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
from vae.vae_base import Sampling
from matplotlib import pyplot as plt

from data_utils import (
    load_yaml_file,
    load_data,
    split_data,
    scale_data,
    inverse_transform_data,
    save_scaler,
    save_data,
)
import paths
from vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)
from visualize import plot_samples, plot_latent_space_samples, visualize_and_save_tsne
vae_type="timeVAE"
appliance = 'load_profile'
data = np.load('../folder/750dataset.npy'.format(appliance, appliance))
data = np.round(data,3)
print("data")
print(data)
# split data into train/valid splits
# train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)
train_data = data
# scale data
# scaled_train_data, scaler = scale_data(train_data)
scaled_train_data = train_data
# ----------------------------------------------------------------------------------
# Instantiate and train the VAE Model

# load hyperparameters from yaml file
hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]
dataset_name = r"C:\Users\ASUS\Desktop\timeVAE-main\outputs\models\sine_subsampled_train_perc_2"

model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
# save scaler
# save_scaler(scaler=scaler, dir_path=model_save_dir)
vae_model = load_vae_model(vae_type, model_save_dir)
# x_decoded = get_posterior_samples(vae_model, scaled_train_data)

prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
# inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
# np.save('{}.npy'.format(appliance), prior_samples)

x_decoded = get_posterior_samples(vae_model, scaled_train_data)
print(x_decoded)

# inverse_scaled_prior_samples = inverse_transform_data(x_decoded, scaler)
np.save('posterior{}.npy'.format("more_epoch50wt3"), x_decoded)
print('posterior{}.npy'.format("more_epoch50wt3"))
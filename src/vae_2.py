import os, warnings
import numpy as np
import time
import matplotlib.pyplot as plt
import visualize


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
from vae.vae_base import Sampling
from matplotlib import pyplot as plt

SEED = 3407
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(SEED)
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

config = {
    "dataset": "washing_machine/washing_machine_cycles",
    "vae_type": "timeVAE",
    "time_step_num": 76
}


def run_vae_pipeline(dataset_name: str, vae_type: str):
    # ----------------------------------------------------------------------------------
    # Load data, perform train/valid split, scale data
    max_epochs = 150

    # read data
    data = load_data(data_dir=paths.DATASETS_DIR, dataset=dataset_name)

    appliance = 'load_profile_more_epoch150_rw3'
    appliance_id = [0]
    # data = np.load(r"C:\Users\ASUS\Desktop\生成模型\timeVAE-main\folder\96dataset.npy") # 682，1440，6
    # data = data[:,:,appliance_id]
    # data = np.transpose(data,(0,2,1))
    print(data.shape)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # fit_transform 对展平后的数据计算均值和标准差，并进行标准化，再次 reshape 回三维结构以适配模型输入要求
    shown_data = data.reshape(-1, 1).reshape(-1, config["time_step_num"], 1)
    visualize.plot_np_array(shown_data, 5, f"origin data {config['dataset']}")
    data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1, config["time_step_num"], 1)
    # split data into train/valid splits
    # train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)
    scaled_train_data = data
    # scale data
    # scaled_train_data, scaler = scale_data(train_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model

    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file(paths.HYPERPARAMETERS_FILE_PATH)[vae_type]
    print(hyperparameters)
    # instantiate the model
    _, sequence_length, feature_dim = scaled_train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )
    print(max_epochs)
    # train vae
    train_vae(
        vae=vae_model,
        train_data=scaled_train_data,
        max_epochs=max_epochs,
        verbose=1,
    )

    # ----------------------------------------------------------------------------------
    # Save scaler and model
    model_save_dir = os.path.join(paths.MODELS_DIR, dataset_name)
    # save scaler
    # save_scaler(scaler=scaler, dir_path=model_save_dir)
    # Save vae
    save_vae_model(vae=vae_model, dir_path=model_save_dir)
    # ----------------------------------------------------------------------------------
    # Visualize posterior samples
    vae_model.summary()

    def get_layer_feature(component, name, data):
        feature_map = tf.keras.models.Model(inputs=component.input, outputs=component.get_layer(name).output)
        enc_conv_1_output = feature_map.predict(x=data, verbose=0)  # 791,22,100
        return enc_conv_1_output

    def plt_data(data):
        for i in range(data.shape[2]):
            plt.subplot(data.shape[2], 1, i + 1)
            plt.plot(data.reshape(data.shape[1], data.shape[2])[:, i], color="blue")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        # plt.title("appliance load")
        plt.show()

    def visualize_hidden_space(vae, scaled_train_data):
        # ------------------ Encoder-------------
        data = scaled_train_data[0].reshape(1, scaled_train_data.shape[1], scaled_train_data.shape[2])  # (1,85,5)
        plt_data(data)
        plt.plot(data.reshape(data.shape[1], data.shape[2]))
        np.save(r'.\outputs\feature_map\input_data.npy', data.reshape(data.shape[1], data.shape[2]))
        plt.show()
        # plt.title("appliance load")
        plt.show()
        encoder_input_fm = get_layer_feature(vae.encoder, 'encoder_input', data)  # (1,85,5)
        enc_conv_1 = get_layer_feature(vae.encoder, 'enc_conv_0', data)  # (1,43,50)
        enc_conv_2 = get_layer_feature(vae.encoder, 'enc_conv_1', data)  # (1,22,100)
        enc_conv_3 = get_layer_feature(vae.encoder, 'enc_conv_2', data).reshape(11, 200)  # (1,11,200)
        enc_flatten = get_layer_feature(vae.encoder, 'enc_flatten', data)  # (1,2000)
        plt.imshow(enc_conv_1.reshape(43, 50))
        plt.title("enc_conv_1")
        np.save(r'.\outputs\feature_map\enc_conv_1.npy', enc_conv_1.reshape(43, 50))

        plt.show()
        plt.imshow(enc_conv_2.reshape(22, 100))
        # plt.title("enc_conv_2")
        np.save(r'.\outputs\feature_map\enc_conv_2.npy', enc_conv_2.reshape(22, 100))

        plt.show()
        plt.imshow(enc_conv_3)
        # plt.title("last_conv")
        plt.show()
        np.save(r'.\outputs\feature_map\enc_conv_3.npy', enc_conv_3)
        z_mean = get_layer_feature(vae.encoder, 'z_mean', data)  # (1,8)
        z_log_var = get_layer_feature(vae.encoder, 'z_log_var', data)  # (1,8)
        encoder_output = Sampling()([z_mean, z_log_var])  # (1,8)
        # ------------------- Decoder level para---------------------------
        level_para_1 = get_layer_feature(vae.decoder, 'level_params1', encoder_output)  # (1,5)
        level_para_2 = get_layer_feature(vae.decoder, 'level_params2', encoder_output)  # (1,5)
        level_params_feature = get_layer_feature(vae.decoder, 'level_params_feature', encoder_output)  # (1,5)
        level_para_final = get_layer_feature(vae.decoder, 'level_param_final', encoder_output)  # (1,85,5)
        plt_data(level_para_final)

        # plt.title("level_params_feature")
        plt.plot(level_para_final.reshape(data.shape[1], data.shape[2]))
        plt.show()
        np.save(r'.\outputs\feature_map\level_para_final.npy', level_para_final)

        # ------------------- Decoder residule---------------------------
        dec_dense = get_layer_feature(vae.decoder, 'dec_dense', encoder_output)  # (1,2000)
        dec_deconv_0 = get_layer_feature(vae.decoder, 'dec_deconv_0', encoder_output)  # (1,22,100)
        dec_deconv_1 = get_layer_feature(vae.decoder, 'dec_deconv_1', encoder_output)  # (1,44,50)
        dec_deconv__2 = get_layer_feature(vae.decoder, 'dec_deconv__2', encoder_output)  # (1,88,5)
        dec_flatten = get_layer_feature(vae.decoder, 'dec_flatten', encoder_output)  # (1,440)
        decoder_dense_final = get_layer_feature(vae.decoder, 'decoder_dense_final', encoder_output)  # (1,425)
        residual_reshape = get_layer_feature(vae.decoder, 'residual_reshape', encoder_output)  # (1,85,5)
        plt.imshow(dec_deconv_0.reshape(22, 100))
        # plt.title("dec_deconv_0")
        plt.show()
        np.save(r'.\feature_map\dec_deconv_0.npy', dec_deconv_0.reshape(22, 100))

        plt.imshow(dec_deconv_1.reshape(44, 50))
        # plt.title("dec_deconv_1")
        plt.show()
        np.save(r'.\outputs\feature_map\dec_deconv_1.npy', dec_deconv_1.reshape(44, 50))

        plt.imshow(dec_deconv__2.reshape(5, 88))
        # plt.title("dec_deconv__2")
        plt.show()
        np.save(r'.\outputs\feature_map\dec_deconv__2.npy', dec_deconv__2.reshape(5, 88))

        plt_data(residual_reshape)
        np.save(r'.\outputs\feature_map\residual_reshape.npy', residual_reshape)
        plt.plot(residual_reshape.reshape(data.shape[1], data.shape[2]))
        plt.show()
        # plt.title("residual_reshape")
        np.save(r'.\outputs\feature_map\residual_reshape.npy', residual_reshape)

        output = vae.predict(data, verbose=0)  # output=data: 791, 85, 5
        plt.plot(output.reshape(85, 5))
        np.save(r'.\outputs\feature_map\output.npy', output.reshape(85, 5))

        plt.show()
        plt_data(output)

    # visualize_hidden_space(vae_model, scaled_train_data)

    # ----------------------------------------------------------------------------------

    x_decoded, mean, variance = get_posterior_samples(vae_model, scaled_train_data)
    x_decoded = scaler.inverse_transform(x_decoded.reshape(-1, 1)).reshape(-1, config["time_step_num"], 1)
    np.save("decoded_samples.npy", x_decoded)
    np.save("variance.npy", variance)
    np.save("mean", mean)
    # plot_samples(
    #     samples1=scaled_train_data,
    #     samples1_name="Original Train",
    #     samples2=x_decoded,
    #     samples2_name="Reconstructed Train",
    #     num_samples=5,
    # )
    # ----------------------------------------------------------------------------------
    # Generate prior samples, visualize and save them

    # Generate prior samples
    prior_samples = get_prior_samples(vae_model, num_samples=1000)
    prior_samples = scaler.inverse_transform(prior_samples.reshape(-1, 1)).reshape(-1, config["time_step_num"], 1)
    np.save('{}.npy'.format("prior_samples"), prior_samples)
    print('{}.npy'.format("prior_samples"))

    # Plot prior samples
    # plot_samples(
    #     samples1=prior_samples,
    #     samples1_name="Prior Samples",
    #     num_samples=5,
    # )

    # visualize t-sne of original and prior samples
    # visualize_and_save_tsne(
    #     samples1=scaled_train_data,
    #     samples1_name="Original",
    #     samples2=prior_samples,
    #     samples2_name="Generated (Prior)",
    #     scenario_name=f"Model-{vae_type} Dataset-{dataset_name}",
    #     save_dir=os.path.join(paths.TSNE_DIR, dataset_name),
    #     max_samples=2000,
    # )

    # inverse transformer samples to original scale and save to dir
    # inverse_scaled_prior_samples = inverse_transform_data(prior_samples, scaler)
    # np.save('{}.npy'.format("prior_samples"),prior_samples)
    # print('{}.npy'.format("prior_samples"))
    # save_data(
    #     data=inverse_scaled_prior_samples,
    #     output_file=os.path.join(
    #         os.path.join(paths.GEN_DATA_DIR, dataset_name),
    #         f"{vae_type}_{dataset_name}_prior_samples.npz",
    #     ),
    # )
    #
    # # ----------------------------------------------------------------------------------
    # # If latent_dim == 2, plot latent space
    # if hyperparameters["latent_dim"] == 2:
    #     plot_latent_space_samples(vae=vae_model, n=8, figsize=(15, 15))
    #
    # # ----------------------------------------------------------------------------------
    # # later.... load model
    # loaded_model = load_vae_model(vae_type, model_save_dir)
    #
    # # Verify that loaded model produces same posterior samples
    # new_x_decoded = loaded_model.predict(scaled_train_data)
    # print(
    #     "Preds from orig and loaded models equal: ",
    #     np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    # )
    # )

    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    # visualize.plot_np_file(f'../data/{config["dataset"]}.npz')
    # check `/data/` for available datasets
    dataset = config["dataset"]
    # models: vae_dense, vae_conv, timeVAE
    model_name = config["vae_type"]

    run_vae_pipeline(dataset, model_name)

    visualize.plot_np_file(r'./prior_samples.npy')

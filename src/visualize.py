import matplotlib.pyplot as plt
import os
import pandas as pd, numpy as np
from sklearn.manifold import TSNE
from typing import Optional

TITLE_FONT_SIZE = 16


def plot_samples(
    samples1: np.ndarray,
    samples1_name: str,
    samples2: Optional[np.ndarray] = None,
    samples2_name: Optional[str] = None,
    num_samples: int = 5,
) -> None:
    """
    Plot one or two sets of samples.

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (Optional[np.ndarray]): The second set of samples to plot.
                                         Defaults to None.
        samples2_name (Optional[str]): The name for the second set of samples in the
                                       plot title.
                                       Defaults to None.
        num_samples (int, optional): The number of samples to plot.
                                     Defaults to 5.

    Returns:
        None
    """
    if samples2 is not None:
        fig, axs = plt.subplots(num_samples, 2, figsize=(10, 6))
    else:
        fig, axs = plt.subplots(num_samples, 1, figsize=(6, 8))

    for i in range(num_samples):
        rnd_idx1 = np.random.choice(len(samples1))
        sample1 = samples1[rnd_idx1]

        if samples2 is not None:
            rnd_idx2 = np.random.choice(len(samples2))
            sample2 = samples2[rnd_idx2]

            axs[i, 0].plot(sample1)
            axs[i, 0].set_title(samples1_name)

            axs[i, 1].plot(sample2)
            axs[i, 1].set_title(samples2_name)
        else:
            axs[i].plot(sample1)
            axs[i].set_title(samples1_name)

    if samples2 is not None:
        fig.suptitle(f"{samples1_name} vs {samples2_name}", fontsize=TITLE_FONT_SIZE)
    else:
        fig.suptitle(samples1_name, fontsize=TITLE_FONT_SIZE)

    fig.tight_layout()
    plt.show()


def plot_latent_space_samples(vae, n: int, figsize: tuple) -> None:
    """
    Plot samples from a 2D latent space.

    Args:
        vae: The VAE model with a method to generate samples from latent space.
        n (int): Number of points in each dimension of the grid.
        figsize (tuple): Figure size for the plot.
    """
    scale = 3.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
    grid_size = len(grid_x)

    # Generate the latent space grid
    Z2 = np.array([[x, y] for x in grid_x for y in grid_y])

    # Generate samples from the VAE given the latent space coordinates
    X_recon = vae.get_prior_samples_given_Z(Z2)
    X_recon = np.squeeze(X_recon)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)

    # Plot each generated sample
    for k, (i, yi) in enumerate(enumerate(grid_y)):
        for j, xi in enumerate(grid_x):
            axs[i, j].plot(X_recon[k])
            axs[i, j].set_title(f"z1={np.round(xi, 2)}; z2={np.round(yi, 2)}")
            k += 1

    fig.suptitle("Generated Samples From 2D Embedded Space", fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()


def avg_over_dim(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Average over the feature dimension of the data.

    Args:
        data (np.ndarray): The data to average over.
        axis (int): Axis to average over.

    Returns:
        np.ndarray: The data averaged over the feature dimension.
    """
    return np.mean(data, axis=axis)


def visualize_and_save_tsne(
    samples1: np.ndarray,
    samples1_name: str,
    samples2: np.ndarray,
    samples2_name: str,
    scenario_name: str,
    save_dir: str,
    max_samples: int = 1000,
) -> None:
    """
    Visualize the t-SNE of two sets of samples and save to file.

    Args:
        samples1 (np.ndarray): The first set of samples to plot.
        samples1_name (str): The name for the first set of samples in the plot title.
        samples2 (np.ndarray): The second set of samples to plot.
        samples2_name (str): The name for the second set of samples in the
                             plot title.
        scenario_name (str): The scenario name for the given samples.
        save_dir (str): Dir path to which to save the file.
        max_samples (int): Maximum number of samples to use in the plot. Samples should
                           be limited because t-SNE is O(n^2).
    """
    if samples1.shape != samples2.shape:
        raise ValueError(
            "Given pairs of samples dont match in shapes. Cannot create t-SNE.\n"
            f"sample1 shape: {samples1.shape}; sample2 shape: {samples2.shape}"
        )

    samples1_2d = avg_over_dim(samples1, axis=2)
    samples2_2d = avg_over_dim(samples2, axis=2)

    # num of samples used in the t-SNE plot
    used_samples = min(samples1_2d.shape[0], max_samples)

    # Combine the original and generated samples
    combined_samples = np.vstack(
        [samples1_2d[:used_samples], samples2_2d[:used_samples]]
    )

    # Compute the t-SNE of the combined samples
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=42)
    tsne_samples = tsne.fit_transform(combined_samples)

    # Create a DataFrame for the t-SNE samples
    tsne_df = pd.DataFrame(
        {
            "tsne_1": tsne_samples[:, 0],
            "tsne_2": tsne_samples[:, 1],
            "sample_type": [samples1_name] * used_samples
            + [samples2_name] * used_samples,
        }
    )

    # Plot the t-SNE samples
    plt.figure(figsize=(8, 8))
    for sample_type, color in zip([samples1_name, samples2_name], ["red", "blue"]):
        if sample_type is not None:
            indices = tsne_df["sample_type"] == sample_type
            plt.scatter(
                tsne_df.loc[indices, "tsne_1"],
                tsne_df.loc[indices, "tsne_2"],
                label=sample_type,
                color=color,
                alpha=0.5,
                s=100,
            )

    plt.title(f"t-SNE for {scenario_name}")
    plt.legend()

    # Save the plot to a file
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{scenario_name}.png"))

    plt.show()


def plot_np_file(file_path: str, num_samples: int = 5, title: str = None, array_key: str = None) -> None:
    """
    专门用于可视化.npy和.npz文件中时间序列数据的函数

    Args:
        file_path (str): .npy或.npz文件路径
        num_samples (int): 要显示的样本数量，默认为5
        title (str): 图表标题，如果未提供则使用文件名
        array_key (str): 当处理.npz文件时，指定要加载的数组键名。如果未提供，使用第一个数组
    """
    # 加载文件
    if file_path.endswith('.npz'):
        loaded = np.load(file_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if array_key is None:
                # 如果没有指定键名，使用第一个键
                array_key = list(loaded.keys())[0]
            data = loaded[array_key]
        else:
            data = loaded
    else:
        data = np.load(file_path)

    # 如果没有提供标题，使用文件名作为标题
    if title is None:
        title = os.path.basename(file_path)

    # 确定要绘制的样本数量
    samples_to_plot = min(num_samples, data.shape[0])

    # 创建子图
    fig, axs = plt.subplots(samples_to_plot, 1, figsize=(10, 2 * samples_to_plot))

    # 如果只有一个样本，将axs转换为列表以便统一处理
    if samples_to_plot == 1:
        axs = [axs]

    # 绘制每个样本
    for i in range(samples_to_plot):
        sample = data[i]
        # 如果是三维数据，展平为一维进行绘制
        if len(sample.shape) > 1:
            sample = sample.flatten()

        axs[i].plot(sample)
        axs[i].set_title(f'{title} - Sample {i + 1}')
        axs[i].grid(True, alpha=0.3)

    fig.suptitle(f'Visualization of {title}', fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()

    # 打印数据基本信息
    print(f"File: {file_path}")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data):.4f}, {np.max(data):.4f}]")
    print(f"Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")

    # 如果是.npz文件，显示所有可用的键
    if file_path.endswith('.npz'):
        loaded = np.load(file_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            print(f"Available keys in .npz file: {list(loaded.keys())}")
            if array_key:
                print(f"Used key: {array_key}")


def plot_np_array(array: np.ndarray, num_samples: int = 5, title: str = None) -> None:
    """
    专门用于可视化numpy数组的函数

    Args:
        array (np.ndarray): 要可视化的numpy数组
        num_samples (int): 要显示的样本数量，默认为5
        title (str): 图表标题，如果未提供则使用文件名
    """
    # 如果没有提供标题，使用文件名作为标题
    if title is None:
        title = "Untitled"

    # 确定要绘制的样本数量
    samples_to_plot = min(num_samples, array.shape[0])

    # 创建子图
    fig, axs = plt.subplots(samples_to_plot, 1, figsize=(10, 2 * samples_to_plot))

    # 如果只有一个样本，将axs转换为列表以便统一处理
    if samples_to_plot == 1:
        axs = [axs]

    # 绘制每个样本
    for i in range(samples_to_plot):
        sample = array[i]
        # 如果是三维数据，展平为一维进行绘制
        if len(sample.shape) > 1:
            sample = sample.flatten()

        axs[i].plot(sample)
        axs[i].set_title(f'{title} - Sample {i + 1}')
        axs[i].grid(True, alpha=0.3)

    fig.suptitle(f'Visualization of {title}', fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    plt.show()

    # 打印数据基本信息
    print(f"Array shape: {array.shape}")
    print(f"Data type: {array.dtype}")
    print(f"Value range: [{np.min(array):.4f}, {np.max(array):.4f}]")
    print(f"Mean: {np.mean(array):.4f}, Std: {np.std(array):.4f}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，解决matplotlib中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_and_extract_similar_cycles(dat_file, output_dir, num_cycles=5):
    """
    分析设备数据，提取相似的工作周期
    """
    print(f"\n🔍 Analyzing: {dat_file}")

    # 读取数据
    timestamps, powers = read_power_data(dat_file)

    if not powers:
        print(f"  ⚠️  File is empty: {dat_file}")
        return None, None

    print(f"  Data points: {len(powers)}")
    print(f"  Power range: {min(powers)} - {max(powers)} W")

    # 识别完整周期
    cycles = identify_complete_cycles(powers, timestamps)

    if not cycles:
        print(f"  ⚠️  No complete cycles found")
        return None, None

    print(f"  Found {len(cycles)} complete cycles")

    # 评估周期质量并选择最佳周期
    best_cycles = select_best_cycles(cycles, num_cycles)

    # 创建输出目录
    device_name = Path(dat_file).stem
    device_output_dir = Path(output_dir) / device_name
    device_output_dir.mkdir(parents=True, exist_ok=True)

    # 保存选定的周期
    save_cycles_as_npz(best_cycles, device_output_dir, device_name)

    return best_cycles, device_name


def read_power_data(dat_file):
    """读取功率数据"""
    timestamps = []
    powers = []

    try:
        with open(dat_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamps.append(int(parts[0]))
                    powers.append(int(parts[1]))
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        with open(dat_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamps.append(int(parts[0]))
                    powers.append(int(parts[1]))

    return timestamps, powers


def extract_pattern_features(power_window):
    """提取功率模式特征"""
    if len(power_window) < 2:
        return None

    # 基本统计特征
    mean_power = np.mean(power_window)
    std_power = np.std(power_window)

    # 趋势特征
    x = np.arange(len(power_window))
    try:
        slope, _, _, _, _ = linregress(x, power_window)
    except:
        slope = 0

    # 峰值特征
    try:
        peaks, _ = find_peaks(power_window, height=max(mean_power * 0.5, 5), distance=5)
        peak_count = len(peaks)
    except:
        peak_count = 0

    # 零功率特征
    zero_ratio = np.sum(np.array(power_window) < 5) / len(power_window)

    # 功率水平特征
    power_levels = identify_power_levels(power_window)

    return {
        'mean': mean_power,
        'std': std_power,
        'slope': slope,
        'peak_count': peak_count,
        'zero_ratio': zero_ratio,
        'power_levels': power_levels
    }


def identify_power_levels(power_window, threshold=10):
    """识别主要的功率水平"""
    # 过滤掉微小波动
    significant_powers = [p for p in power_window if p >= threshold]
    if not significant_powers:
        return []

    # 使用K-means识别功率水平
    if len(significant_powers) >= 3:
        power_array = np.array(significant_powers).reshape(-1, 1)
        n_clusters = min(3, len(set(significant_powers)))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(power_array)
            return sorted([int(center[0]) for center in kmeans.cluster_centers_])

    return [int(np.mean(significant_powers))]


def identify_working_modes(power_series, min_pattern_length=30):
    """识别工作模式"""
    patterns = []
    window_size = min_pattern_length
    step_size = max(window_size // 3, 1)  # 重叠窗口，确保至少为1

    for i in range(0, len(power_series) - window_size + 1, step_size):
        window = power_series[i:i + window_size]

        # 只处理包含显著功率的窗口
        if np.max(window) < 10:  # 忽略低功率窗口
            continue

        features = extract_pattern_features(window)
        if features:
            patterns.append({
                'start_idx': i,
                'end_idx': i + window_size,
                'features': features,
                'data': window
            })

    return patterns


def cluster_similar_patterns(patterns, n_clusters=5):
    """聚类相似模式"""
    if len(patterns) < n_clusters:
        n_clusters = max(1, len(patterns))

    feature_vectors = []
    for pattern in patterns:
        features = pattern['features']
        vector = [
            features['mean'],
            features['std'],
            features['slope'],
            features['peak_count'],
            features['zero_ratio']
        ]
        # 添加功率水平特征
        if features['power_levels']:
            # 确保向量长度一致
            for level in features['power_levels'][:3]:  # 最多取3个功率水平
                vector.append(level)
            # 如果功率水平不足3个，填充0
            while len(vector) < 8:  # 5个基础特征 + 最多3个功率水平
                vector.append(0)
        else:
            vector.extend([0, 0, 0])  # 填充3个0

    feature_vectors.append(vector)

    # 确保所有向量长度一致
    max_len = max(len(v) for v in feature_vectors)
    for i, v in enumerate(feature_vectors):
        if len(v) < max_len:
            feature_vectors[i] = v + [0] * (max_len - len(v))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)

    for i, pattern in enumerate(patterns):
        pattern['cluster'] = labels[i]

    return patterns, kmeans


def merge_patterns_to_cycle(patterns, timestamps):
    """将连续相似模式合并为完整周期"""
    start_idx = patterns[0]['start_idx']
    end_idx = patterns[-1]['end_idx']

    # 去除重复部分（由于重叠窗口）
    unique_powers = []
    current_idx = start_idx
    for pattern in patterns:
        pattern_start = pattern['start_idx']
        pattern_end = pattern['end_idx']

        if pattern_start >= current_idx:
            # 添加非重叠部分
            if pattern_start > current_idx:
                # 添加间隙数据（如果有）
                pass
            unique_powers.extend(pattern['data'])
            current_idx = pattern_end

    # 如果合并后数据太长，适当截断
    if len(unique_powers) > 1000:
        unique_powers = unique_powers[:1000]
        end_idx = start_idx + 1000

    return {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_time': timestamps[start_idx],
        'end_time': timestamps[min(end_idx, len(timestamps) - 1)],
        'duration': timestamps[min(end_idx, len(timestamps) - 1)] - timestamps[start_idx],
        'timesteps': len(unique_powers),
        'power_values': unique_powers,
        'patterns': patterns,
        'cluster': patterns[0]['cluster'],
        'pattern_count': len(patterns)
    }


def identify_complete_cycles(power_data, timestamps):
    """识别完整工作周期"""
    patterns = identify_working_modes(power_data)

    if len(patterns) < 3:  # 至少需要3个模式才能有效识别周期
        return []

    patterns, kmeans = cluster_similar_patterns(patterns)

    cycles = []
    current_cycle = []
    current_cluster = None

    for pattern in patterns:
        if current_cluster is None:
            current_cluster = pattern['cluster']
            current_cycle.append(pattern)
        elif pattern['cluster'] == current_cluster:
            # 检查连续性（模式之间不能有太大间隔）
            if pattern['start_idx'] - current_cycle[-1]['end_idx'] <= 100:  # 最大间隔
                current_cycle.append(pattern)
            else:
                # 间隔太大，结束当前周期
                if len(current_cycle) >= 2:
                    cycle = merge_patterns_to_cycle(current_cycle, timestamps)
                    if cycle['timesteps'] >= 50:  # 确保周期足够长
                        cycles.append(cycle)
                current_cluster = pattern['cluster']
                current_cycle = [pattern]
        else:
            # 簇变化，结束当前周期
            if len(current_cycle) >= 2:
                cycle = merge_patterns_to_cycle(current_cycle, timestamps)
                if cycle['timesteps'] >= 50:  # 确保周期足够长
                    cycles.append(cycle)
            current_cluster = pattern['cluster']
            current_cycle = [pattern]

    if len(current_cycle) >= 2:
        cycle = merge_patterns_to_cycle(current_cycle, timestamps)
        if cycle['timesteps'] >= 50:  # 确保周期足够长
            cycles.append(cycle)

    return cycles


def calculate_pattern_similarity(patterns):
    """计算模式相似度"""
    if len(patterns) < 2:
        return 0

    similarities = []
    for i in range(len(patterns) - 1):
        feat1 = patterns[i]['features']
        feat2 = patterns[i + 1]['features']

        # 计算特征相似度
        mean_diff = abs(feat1['mean'] - feat2['mean'])
        mean_sim = 1 - mean_diff / max(feat1['mean'], feat2['mean'], 1)

        std_diff = abs(feat1['std'] - feat2['std'])
        std_sim = 1 - std_diff / max(feat1['std'], feat2['std'], 1)

        similarities.append((mean_sim + std_sim) / 2)

    return np.mean(similarities)


def calculate_regularity(cycle):
    """计算周期规律性"""
    patterns = cycle['patterns']
    if len(patterns) < 3:
        return 0

    # 计算模式持续时间的规律性
    durations = [p['end_idx'] - p['start_idx'] for p in patterns]
    duration_std = np.std(durations)
    max_duration = max(durations)
    regularity = 1 - (duration_std / max_duration) if max_duration > 0 else 0

    return regularity


def select_best_cycles(cycles, num_cycles):
    """选择最佳周期"""
    scored_cycles = []

    for cycle in cycles:
        pattern_similarity = calculate_pattern_similarity(cycle['patterns'])
        regularity = calculate_regularity(cycle)

        # 周期长度也是一个考量因素
        length_score = min(1.0, cycle['timesteps'] / 200)  # 偏好较长周期

        quality_score = (pattern_similarity * 0.4 +
                         regularity * 0.3 +
                         length_score * 0.3)

        cycle['quality_score'] = quality_score
        cycle['pattern_similarity'] = pattern_similarity
        cycle['regularity'] = regularity

        scored_cycles.append(cycle)

    # 按质量排序并选择最佳周期
    best_cycles = sorted(scored_cycles, key=lambda x: x['quality_score'], reverse=True)[:num_cycles]

    print(f"  Selected {len(best_cycles)} best cycles:")
    for i, cycle in enumerate(best_cycles):
        print(f"    Cycle {i + 1}: {cycle['timesteps']} steps, "
              f"{cycle['pattern_count']} patterns, "
              f"quality: {cycle['quality_score']:.3f}")

    return best_cycles


def save_cycles_as_npz(cycles, output_dir, device_name):
    """保存周期数据"""
    if not cycles:
        return None

    # 将所有周期数据组合成一个numpy数组 (n_cycles, timesteps, 1)
    all_cycles_data = []
    max_timesteps = max(cycle['timesteps'] for cycle in cycles)

    for i, cycle in enumerate(cycles):
        power_data = np.array(cycle['power_values'])

        if len(power_data) < max_timesteps:
            padded_data = np.pad(power_data, (0, max_timesteps - len(power_data)),
                                 mode='constant', constant_values=0)
            all_cycles_data.append(padded_data)
        else:
            all_cycles_data.append(power_data)

    data_array = np.array(all_cycles_data).reshape(len(cycles), max_timesteps, 1)

    # 保存文件
    filename = f"{device_name}_similar_cycles.npz"
    filepath = output_dir / filename
    np.savez(filepath, data=data_array)

    print(f"  💾 Saved: {filename} (shape: {data_array.shape})")

    # 保存周期信息
    cycle_info = {
        'device': device_name,
        'total_cycles': len(cycles),
        'timesteps_per_cycle': max_timesteps,
        'cycle_timesteps': [cycle['timesteps'] for cycle in cycles],
        'pattern_counts': [cycle['pattern_count'] for cycle in cycles],
        'quality_scores': [cycle['quality_score'] for cycle in cycles],
        'pattern_similarities': [cycle['pattern_similarity'] for cycle in cycles]
    }

    info_file = output_dir / "cycle_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Device: {cycle_info['device']}\n")
        f.write(f"Total cycles: {cycle_info['total_cycles']}\n")
        f.write(f"Timesteps per cycle: {cycle_info['timesteps_per_cycle']}\n")
        f.write(f"Original cycle timesteps: {cycle_info['cycle_timesteps']}\n")
        f.write(f"Pattern counts: {cycle_info['pattern_counts']}\n")
        f.write(f"Quality scores: {cycle_info['quality_scores']}\n")
        f.write(f"Pattern similarities: {cycle_info['pattern_similarities']}\n")
        f.write(f"\nNPZ file shape: {data_array.shape}\n")

    return cycle_info


def generate_summary_report(all_cycles_info, output_dir):
    """
    生成汇总报告
    """
    print(f"\n{'=' * 60}")
    print("📊 Processing Complete - Summary Report")
    print(f"{'=' * 60}")

    summary_data = []

    for device_name, cycles_info in all_cycles_info.items():
        if cycles_info:
            summary_data.append({
                'Device': device_name,
                'Cycles': cycles_info['total_cycles'],
                'Timesteps': cycles_info['timesteps_per_cycle'],
                'Original Timesteps': cycles_info['cycle_timesteps'],
                'Pattern Counts': cycles_info['pattern_counts'],
                'Quality Scores': [f"{score:.3f}" for score in cycles_info['quality_scores']],
                'Pattern Similarities': [f"{sim:.3f}" for sim in cycles_info['pattern_similarities']]
            })

            print(f"\n{device_name}:")
            print(f"  Extracted cycles: {cycles_info['total_cycles']}")
            print(f"  Final timesteps: {cycles_info['timesteps_per_cycle']}")
            print(f"  Original timesteps: {cycles_info['cycle_timesteps']}")
            print(f"  Pattern counts: {cycles_info['pattern_counts']}")
            print(f"  Quality scores: {cycles_info['quality_scores']}")
            print(f"  Pattern similarities: {cycles_info['pattern_similarities']}")

    # 保存汇总报告 - 使用UTF-8编码
    summary_file = Path(output_dir) / "processing_summary.csv"
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False, encoding='utf-8-sig')  # 使用UTF-8带BOM编码
    print(f"\n💾 Summary report saved: {summary_file}")


def plot_sample_segments(all_cycles_info, output_dir):
    """
    绘制样本片段的功率曲线 - 使用英文标签避免字体问题
    """
    # 为每个设备创建一个图表
    for device_name, cycles_info in all_cycles_info.items():
        if not cycles_info:
            continue

        # 加载数据
        data_file = Path(output_dir) / device_name / f"{device_name}_similar_cycles.npz"
        if data_file.exists():
            data = np.load(data_file)
            all_cycles = data['data']

            # 创建图表
            n_cycles = min(5, len(all_cycles))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            # 绘制前几个周期
            for i in range(n_cycles):
                if i < len(axes):
                    power_series = all_cycles[i, :, 0]
                    time_axis = np.arange(len(power_series))

                    # 只绘制非零部分
                    nonzero_indices = np.where(power_series > 0)[0]
                    if len(nonzero_indices) > 0:
                        start_idx = max(0, nonzero_indices[0] - 5)
                        end_idx = min(len(power_series), nonzero_indices[-1] + 5)
                        plotted_power = power_series[start_idx:end_idx]
                        plotted_time = time_axis[start_idx:end_idx]
                    else:
                        plotted_power = power_series
                        plotted_time = time_axis

                    axes[i].plot(plotted_time, plotted_power, 'b-', linewidth=1.5)
                    axes[i].set_title(f'{device_name} - Cycle {i + 1}\n{len(power_series)} steps')
                    axes[i].set_xlabel('Time Steps')
                    axes[i].set_ylabel('Power (W)')
                    axes[i].grid(True, alpha=0.3)

            # 隐藏多余的子图
            for i in range(n_cycles, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(Path(output_dir) / device_name / f"{device_name}_cycles_plot.png",
                        dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免显示

    # 创建一个汇总图表，显示每个设备的第一个周期
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    device_count = 0
    for device_name, cycles_info in all_cycles_info.items():
        if device_count >= len(axes) or not cycles_info:
            continue

        # 加载数据
        data_file = Path(output_dir) / device_name / f"{device_name}_similar_cycles.npz"
        if data_file.exists():
            data = np.load(data_file)
            first_cycle = data['data'][0, :, 0]

            # 只绘制非零部分
            nonzero_indices = np.where(first_cycle > 0)[0]
            if len(nonzero_indices) > 0:
                start_idx = max(0, nonzero_indices[0] - 5)
                end_idx = min(len(first_cycle), nonzero_indices[-1] + 5)
                plotted_power = first_cycle[start_idx:end_idx]
                plotted_time = np.arange(len(plotted_power))
            else:
                plotted_power = first_cycle
                plotted_time = np.arange(len(first_cycle))

            ax = axes[device_count]
            ax.plot(plotted_time, plotted_power, 'b-', linewidth=1.5)
            ax.set_title(f'{device_name}\n{len(first_cycle)} steps')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Power (W)')
            ax.grid(True, alpha=0.3)

            device_count += 1

    # 隐藏多余的子图
    for i in range(device_count, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "all_devices_first_cycle.png", dpi=300, bbox_inches='tight')
    plt.show()


def fix_existing_encoding_issues():
    """
    修复已存在的编码问题文件
    """
    processed_dir = Path("./processed_data")

    if not processed_dir.exists():
        return

    # 查找所有cycle_info.txt文件并重新保存为UTF-8
    for info_file in processed_dir.glob("**/cycle_info.txt"):
        try:
            # 尝试用不同编码读取
            with open(info_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 用UTF-8重新写入
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ Fixed encoding: {info_file}")

        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试其他编码
            for encoding in ['gbk', 'latin-1', 'cp1252']:
                try:
                    with open(info_file, 'r', encoding=encoding) as f:
                        content = f.read()

                    with open(info_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    print(f"✅ Fixed encoding ({encoding} -> UTF-8): {info_file}")
                    break
                except:
                    continue


def main():
    """
    主函数：处理所有设备数据
    """
    # 配置路径
    data_dir = "./pre_data"  # 原始数据目录
    output_dir = "./data"  # 处理后的数据目录

    # 设备文件列表
    devices = [
        "fridge.dat",
        "washing_machine.dat",
    ]

    print("🚀 Starting to process all device data...")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 设备特定的参数（可根据实际情况调整）
    device_params = {
        "laptop": {"min_pattern_length": 20, "num_cycles": 5},
        "kettle": {"min_pattern_length": 10, "num_cycles": 5},
        "microwave": {"min_pattern_length": 15, "num_cycles": 5},
        "washing_machine": {"min_pattern_length": 30, "num_cycles": 5},
        "dishwasher": {"min_pattern_length": 25, "num_cycles": 5}
    }

    all_cycles_info = {}

    # 处理每个设备
    for device_file in devices:
        device_path = Path(data_dir) / device_file

        if not device_path.exists():
            print(f"⚠️  Skipping: {device_path} does not exist")
            continue

        device_name = device_path.stem
        params = device_params.get(device_name, {"min_pattern_length": 25, "num_cycles": 5})

        # 这里简化调用，实际使用时可以根据params调整算法参数
        cycles, device_name = analyze_and_extract_similar_cycles(
            device_path,
            output_dir,
            num_cycles=params["num_cycles"]
        )

        if cycles:
            # 保存周期信息
            cycle_info = {
                'total_cycles': len(cycles),
                'timesteps_per_cycle': max(cycle['timesteps'] for cycle in cycles),
                'cycle_timesteps': [cycle['timesteps'] for cycle in cycles],
                'pattern_counts': [cycle['pattern_count'] for cycle in cycles],
                'quality_scores': [cycle['quality_score'] for cycle in cycles],
                'pattern_similarities': [cycle['pattern_similarity'] for cycle in cycles]
            }
            all_cycles_info[device_name] = cycle_info

    # 生成报告和图表
    generate_summary_report(all_cycles_info, output_dir)
    plot_sample_segments(all_cycles_info, output_dir)

    # 修复可能存在的编码问题
    fix_existing_encoding_issues()

    print(f"\n🎉 All processing completed!")
    print(f"📁 Data saved in: {output_dir}")
    print(f"📊 Each device has 5 similar cycles in a single NPZ file")
    print(f"📋 Summary report: {output_dir}/processing_summary.csv")
    print(f"🖼️  Sample charts: {output_dir}/[device_name]/[device_name]_cycles_plot.png")


if __name__ == "__main__":
    main()
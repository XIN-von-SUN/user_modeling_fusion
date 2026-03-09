"""
Gaze Data Visualization Module

This module provides comprehensive visualization functions for eye-tracking gaze data analysis.

Main features:
- Gaze heatmap visualization with AOI boundaries
- Saccade trajectory visualization
- Detailed saccade-event analysis
- Improved trajectory visualization

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def visualize_gaze_heatmap_by_trial(data, participant_id, response_data=None, max_trials=None,
                                   figsize_per_trial=(4, 4), show_aoi_boundaries=True, verbose=True,
                                   style='academic', use_original_aoi=True):
    """
    Visualize gaze-point distributions as heatmaps for individual trials in the
    stimuli coordinate system `[-1, 1]`.

    Parameters:
    - data: Integrated gaze data (DataFrame)
    - participant_id: Participant ID (str)
    - response_data: Optional response table for `reasoning_type`
    - max_trials: Maximum number of trials to display; `None` shows all
    - figsize_per_trial: Size of each subplot
    - show_aoi_boundaries: Whether to draw AOI boundaries
    - verbose: Whether to print detailed statistics
    - style: Visualization style (`academic` or `modern`)
    - use_original_aoi: `True` for the original 5 AOIs, `False` for the merged 3 AOIs

    Returns:
    - None (plots the figure)
    """
    
    # Define style settings.
    if style == 'academic':
        # Academic-journal style with a white background.
        fig_bg = 'white'
        ax_bg = 'white'
        text_color = 'black'
        grid_color = '#CCCCCC'
        grid_alpha = 0.5
        cross_color = '#666666'
        cross_style = '--'
        cross_alpha = 0.6
        # Academic-style heatmap colors.
        heatmap_colors = ['#FFFFFF', '#FFF2CC', '#FFE599', '#FFD966', '#FF9900', '#E69138', '#CC6600', '#990000']
        custom_cmap = LinearSegmentedColormap.from_list('academic_heatmap', heatmap_colors, N=256)
        # Academic-style AOI colors.
        if use_original_aoi:
            aoi_colors = {
                'claim_aoi': '#DC3545',      # Dark red
                'evidence_aoi': '#17A2B8',   # Dark cyan
                'answer_aoi': '#0056B3',     # Dark blue
                'reasoning_aoi': '#28A745',  # Dark green
                'response_aoi': '#FD7E14'    # Dark orange
            }
        else:
            # Colors for merged AOIs.
            aoi_colors = {
                'claim_evidence': '#DC3545',     # Dark red - Context
                'answer_reasoning': '#0056B3',   # Dark blue - AI response
                'response_aoi': '#FD7E14'        # Dark orange - Rating
            }
        aoi_alpha = 0.05  # Lighter fill.
        label_bg = 'white'
        label_alpha = 0.9
    else:  # modern style
        # Modern dark-theme style.
        fig_bg = '#34495E'
        ax_bg = '#2C3E50'
        text_color = 'white'
        grid_color = '#BDC3C7'
        grid_alpha = 0.4
        cross_color = '#E74C3C'
        cross_style = '-'
        cross_alpha = 0.7
        # Modern-style heatmap colors.
        heatmap_colors = ['#1A1A2E', '#16213E', '#0F3460', '#E94560', '#F39C12', '#F1C40F', '#FFFFFF']
        custom_cmap = LinearSegmentedColormap.from_list('modern_heatmap', heatmap_colors, N=256)
        # Modern-style AOI colors.
        if use_original_aoi:
            aoi_colors = {
                'claim_aoi': '#FF4757',      # Bright red
                'evidence_aoi': '#00D2D3',   # Bright cyan
                'answer_aoi': '#5352ED',     # Bright blue
                'reasoning_aoi': '#2ED573',  # Bright green
                'response_aoi': '#FFA502'    # Bright orange
            }
        else:
            # Colors for merged AOIs.
            aoi_colors = {
                'claim_evidence': '#FF4757',     # Bright red - Context
                'answer_reasoning': '#5352ED',   # Bright blue - AI response
                'response_aoi': '#FFA502'        # Bright orange - Rating
            }
        aoi_alpha = 0.15  # Darker fill.
        label_bg = 'black'
        label_alpha = 0.8
    
    # Filter rows with valid coordinates.
    valid_data = data.dropna(subset=['stimuli_x', 'stimuli_y', 'trial_num']).copy()
    
    if len(valid_data) == 0:
        print("❌ No valid coordinate data found!")
        return
    
    # If response_data is provided, build a trial-info lookup.
    trial_info = {}
    if response_data is not None:
        for _, row in response_data.iterrows():
            trial_info[row['trial_num']] = {
                'condition': row['condition'],
                'reasoning_type': row.get('reasoning_type', 'unknown')
            }
    
    # Collect all trial IDs.
    trials = sorted(valid_data['trial_num'].unique())
    if max_trials:
        trials = trials[:max_trials]
    
    if verbose:
        print(f"📊 Creating heatmaps for {len(trials)} trials for Participant {participant_id}")
        print(f"   Valid gaze points: {len(valid_data)}")
        print(f"   Coordinate range: X[{valid_data['stimuli_x'].min():.3f}, {valid_data['stimuli_x'].max():.3f}], Y[{valid_data['stimuli_y'].min():.3f}, {valid_data['stimuli_y'].max():.3f}]")
    
    # Compute subplot layout.
    n_trials = len(trials)
    n_cols = min(4, n_trials)  # At most 4 columns.
    n_rows = (n_trials + n_cols - 1) // n_cols  # Round up.
    
    # Create the figure.
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_trial[0] * n_cols, figsize_per_trial[1] * n_rows),
                            facecolor=fig_bg)  # Use the style-specific figure background.
    
    # Apply global figure styling.
    fig.patch.set_facecolor(fig_bg)
    
    # If there is only one subplot, `axes` is not an array.
    if n_trials == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create one heatmap per trial.
    for idx, trial_num in enumerate(trials):
        row = idx // n_cols
        col = idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Get data for the current trial.
        trial_data = valid_data[valid_data['trial_num'] == trial_num]
        
        if len(trial_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Trial {int(trial_num)}')
            continue
        
        # Get condition and reasoning_type for the current trial.
        condition = trial_data['condition'].iloc[0] if 'condition' in trial_data.columns else 'unknown'
        reasoning_type = 'unknown'
        if trial_num in trial_info:
            reasoning_type = trial_info[trial_num]['reasoning_type']
        
        # Build a title that includes condition and reasoning type.
        title_parts = [f'Trial {int(trial_num)}']
        title_parts.append(f'({len(trial_data)} pts)')
        
        # 添加condition信息（简化显示）
        condition_short = condition.replace('_', ' ').replace('reasoning', 'reas').replace('without', 'w/o').replace('with', 'w/')
        title_parts.append(f'{condition_short}')
        
        # 添加reasoning type信息（如果不是none或unknown）
        if reasoning_type not in ['none', 'unknown']:
            reasoning_short = reasoning_type.replace('_', ' ')
            title_parts.append(f'{reasoning_short}')
        
        title_text = '\n'.join([title_parts[0] + ' ' + title_parts[1], ' '.join(title_parts[2:])])
        
        # 创建2D直方图用于热力图
        # 设置bins范围为[-1, 1]
        x_data = trial_data['stimuli_x'].values
        y_data = trial_data['stimuli_y'].values
        
        # 设置图形背景色
        ax.set_facecolor(ax_bg)
        
        # 创建热力图
        hb = ax.hexbin(x_data, y_data, 
                      gridsize=35,  # 稍微增加精度
                      extent=[-1, 1, -1, 1],  # 固定范围为[-1, 1]
                      cmap=custom_cmap,
                      alpha=0.9,  # 增加不透明度
                      mincnt=1)  # 至少1个点才显示
        
        # 设置坐标轴
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Stimuli X', color=text_color, fontweight='bold')
        ax.set_ylabel('Stimuli Y', color=text_color, fontweight='bold')
        ax.set_title(title_text, fontsize=10, pad=10, color=text_color, fontweight='bold')
        
        # 设置坐标轴文字颜色
        ax.tick_params(colors=text_color, which='both')
        
        # 添加中心十字线
        ax.axhline(y=0, color=cross_color, linestyle=cross_style, alpha=cross_alpha, linewidth=1.5)
        ax.axvline(x=0, color=cross_color, linestyle=cross_style, alpha=cross_alpha, linewidth=1.5)
        
        # 添加网格
        ax.grid(True, alpha=grid_alpha, color=grid_color, linestyle=':')
        
        # 设置刻度
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        
        # 添加颜色条
        plt.colorbar(hb, ax=ax, shrink=0.8)
        
        # 添加AOI边界线
        if show_aoi_boundaries and 'aoi_left' in trial_data.columns:
            # 获取该trial的唯一AOI边界
            aoi_boundaries = trial_data[trial_data['aoi'] != 'OOD'].copy()
            if len(aoi_boundaries) > 0:
                if use_original_aoi:
                    # 使用原始AOI
                    unique_aois = aoi_boundaries.groupby(['aoi', 'aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']).size().reset_index()

                    for _, aoi_row in unique_aois.iterrows():
                        aoi_name = aoi_row['aoi']
                        left, right = aoi_row['aoi_left'], aoi_row['aoi_right']
                        top, bottom = aoi_row['aoi_top'], aoi_row['aoi_bottom']

                        # 确保边界在显示范围内
                        left = max(left, -1); right = min(right, 1)
                        top = min(top, 1); bottom = max(bottom, -1)

                        color = aoi_colors.get(aoi_name, text_color)

                        # Draw the AOI boundary rectangle.
                        rect = Rectangle((left, bottom), right-left, top-bottom,
                                       linewidth=3, edgecolor=color, facecolor='none',
                                       alpha=0.9, linestyle='-')
                        ax.add_patch(rect)

                        # Add a translucent fill to make the region easier to identify.
                        fill_rect = Rectangle((left, bottom), right-left, top-bottom,
                                            linewidth=0, edgecolor='none', facecolor=color,
                                            alpha=aoi_alpha)
                        ax.add_patch(fill_rect)

                        # Add the AOI label.
                        label_x = left + (right - left) * 0.05
                        label_y = top - (top - bottom) * 0.1
                        ax.text(label_x, label_y, aoi_name.replace('_aoi', '').upper(),
                               fontsize=9, fontweight='bold', color=color,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg,
                                       edgecolor=color, alpha=label_alpha, linewidth=1))
                else:
                    # 使用合并的AOI
                    # 先获取所有原始AOI的边界
                    aoi_groups = {}
                    for _, aoi_row in aoi_boundaries.groupby(['aoi', 'aoi_left', 'aoi_right', 'aoi_top', 'aoi_bottom']).size().reset_index().iterrows():
                        aoi_name = aoi_row['aoi']
                        if aoi_name in ['claim_aoi', 'evidence_aoi']:
                            group_name = 'claim_evidence'
                        elif aoi_name in ['answer_aoi', 'reasoning_aoi']:
                            group_name = 'answer_reasoning'
                        elif aoi_name == 'response_aoi':
                            group_name = 'response_aoi'
                        else:
                            continue

                        if group_name not in aoi_groups:
                            aoi_groups[group_name] = {
                                'left': aoi_row['aoi_left'],
                                'right': aoi_row['aoi_right'],
                                'top': aoi_row['aoi_top'],
                                'bottom': aoi_row['aoi_bottom']
                            }
                        else:
                            # 合并边界
                            aoi_groups[group_name]['left'] = min(aoi_groups[group_name]['left'], aoi_row['aoi_left'])
                            aoi_groups[group_name]['right'] = max(aoi_groups[group_name]['right'], aoi_row['aoi_right'])
                            aoi_groups[group_name]['top'] = max(aoi_groups[group_name]['top'], aoi_row['aoi_top'])
                            aoi_groups[group_name]['bottom'] = min(aoi_groups[group_name]['bottom'], aoi_row['aoi_bottom'])

                    # 绘制合并后的AOI
                    aoi_labels = {
                        'claim_evidence': 'CONTEXT',
                        'answer_reasoning': 'AI RESPONSE',
                        'response_aoi': 'RATING'
                    }

                    for group_name, bounds in aoi_groups.items():
                        left = max(bounds['left'], -1)
                        right = min(bounds['right'], 1)
                        top = min(bounds['top'], 1)
                        bottom = max(bounds['bottom'], -1)

                        color = aoi_colors.get(group_name, text_color)

                        # 绘制AOI边界矩形
                        rect = Rectangle((left, bottom), right-left, top-bottom,
                                       linewidth=3, edgecolor=color, facecolor='none',
                                       alpha=0.9, linestyle='-')
                        ax.add_patch(rect)

                        # 添加半透明填充以增强区域识别
                        fill_rect = Rectangle((left, bottom), right-left, top-bottom,
                                            linewidth=0, edgecolor='none', facecolor=color,
                                            alpha=aoi_alpha)
                        ax.add_patch(fill_rect)

                        # 添加AOI标签
                        label_x = left + (right - left) * 0.05
                        label_y = top - (top - bottom) * 0.1
                        label_text = aoi_labels.get(group_name, group_name.upper())
                        ax.text(label_x, label_y, label_text,
                               fontsize=9, fontweight='bold', color=color,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg,
                                       edgecolor=color, alpha=label_alpha, linewidth=1))
    
    # Hide unused subplots.
    for idx in range(n_trials, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Gaze Heatmaps by Trial - Participant {participant_id}', y=1.02, fontsize=16)
    plt.show()
    
    # Print summary statistics.
    if verbose:
        print(f"\n📈 Trial Statistics:")
        for trial_num in trials:
            trial_data = valid_data[valid_data['trial_num'] == trial_num]
            if len(trial_data) > 0:
                condition = trial_data['condition'].iloc[0] if 'condition' in trial_data.columns else 'unknown'
                reasoning_type = trial_info.get(trial_num, {}).get('reasoning_type', 'unknown') if trial_info else 'unknown'
                
                print(f"   Trial {int(trial_num)}: {len(trial_data)} points, "
                      f"Condition: {condition}, Reasoning: {reasoning_type}, "
                      f"X_range[{trial_data['stimuli_x'].min():.2f}, {trial_data['stimuli_x'].max():.2f}], "
                      f"Y_range[{trial_data['stimuli_y'].min():.2f}, {trial_data['stimuli_y'].max():.2f}]")


def visualize_saccade_trajectories_by_stimulus(data, participant_id, max_stimuli=None, 
                                              figsize_per_stimulus=(5, 5), verbose=True):
    """
    Visualize saccade trajectories for each stimulus.
    
    Parameters:
    - data: Integrated gaze data (DataFrame)
    - participant_id: Participant ID (str)
    - max_stimuli: Maximum number of stimuli to show; `None` shows all
    - figsize_per_stimulus: Size of each subplot
    - verbose: Whether to print detailed statistics
    
    Returns:
    - None (plots the figure)
    """
    
    # Filter valid saccade events and coordinates.
    saccade_data = data[(data['event_type'] == 'saccade')].copy()
    valid_saccades = saccade_data.dropna(subset=['stimuli_x', 'stimuli_y', 'trial_num', 'device_time_stamp']).copy()
    
    if len(valid_saccades) == 0:
        print("❌ No valid saccade-trajectory data found")
        return
    
    # Group by `trial_num` (each trial corresponds to one stimulus).
    trials = sorted(valid_saccades['trial_num'].unique())
    if max_stimuli:
        trials = trials[:max_stimuli]
    
    if verbose:
        print(f"📊 创建 {len(trials)} 个 Stimulus 的 Saccade 轨迹图")
        print(f"   总 Saccade 点数: {len(valid_saccades)}")
    
    # Compute subplot layout.
    n_trials = len(trials)
    n_cols = min(3, n_trials)  # 最多3列
    n_rows = (n_trials + n_cols - 1) // n_cols  # 向上取整
    
    # Create the figure.
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_stimulus[0] * n_cols, figsize_per_stimulus[1] * n_rows))
    
    # Handle the single-subplot case.
    if n_trials == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Create one trajectory plot per trial.
    for idx, trial_num in enumerate(trials):
        row = idx // n_cols
        col = idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # Get saccade data for the current trial.
        trial_saccades = valid_saccades[valid_saccades['trial_num'] == trial_num].copy()
        
        if len(trial_saccades) == 0:
            ax.text(0.5, 0.5, 'No Saccade Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Trial {int(trial_num)} - No Saccades')
            continue
        
        # Sort by time.
        trial_saccades = trial_saccades.sort_values('device_time_stamp')
        
        # Get coordinates.
        x_coords = trial_saccades['stimuli_x'].values
        y_coords = trial_saccades['stimuli_y'].values
        timestamps = trial_saccades['device_time_stamp'].values
        
        # Draw the trajectory line.
        ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1, label='Saccade Path')
        
        # Draw the start point (green).
        ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', 
                  label='Start', zorder=5, edgecolor='black', linewidth=1)
        
        # Draw the end point (red).
        ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='s', 
                  label='End', zorder=5, edgecolor='black', linewidth=1)
        
        # Draw intermediate points with time-based color encoding.
        if len(x_coords) > 2:
            # Create a color map over time progression.
            time_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
            scatter = ax.scatter(x_coords[1:-1], y_coords[1:-1], 
                               c=time_normalized[1:-1], cmap='viridis', 
                               s=30, alpha=0.7, zorder=3)
            
            # Add a color bar.
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Time Progress', fontsize=8)
        
        # Add direction arrows at regular intervals.
        arrow_step = max(1, len(x_coords) // 8)  # 最多显示8个箭头
        for i in range(0, len(x_coords) - 1, arrow_step):
            if i + 1 < len(x_coords):
                dx = x_coords[i + 1] - x_coords[i]
                dy = y_coords[i + 1] - y_coords[i]
                # Only draw arrows when the movement is large enough.
                if np.sqrt(dx**2 + dy**2) > 0.05:
                    ax.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]), 
                              xytext=(x_coords[i], y_coords[i]),
                              arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7, lw=1))
        
        # Configure axes.
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Stimuli X')
        ax.set_ylabel('Stimuli Y')
        
        # Get trial metadata.
        condition = 'unknown'
        stimulus_id = 'unknown'
        if 'condition' in trial_saccades.columns:
            condition = trial_saccades['condition'].iloc[0] if len(trial_saccades) > 0 else 'unknown'
        if 'stimulus_id' in trial_saccades.columns:
            stimulus_id = trial_saccades['stimulus_id'].iloc[0] if len(trial_saccades) > 0 else 'unknown'
        
        # Build the title.
        title_parts = [f'Trial {int(trial_num)}']
        if stimulus_id != 'unknown':
            title_parts.append(f'(Stimulus {stimulus_id})')
        title_parts.append(f'{len(trial_saccades)} saccades')
        
        condition_short = str(condition).replace('_', ' ').replace('reasoning', 'reas')
        title_text = '\n'.join([' '.join(title_parts), condition_short])
        
        ax.set_title(title_text, fontsize=10, pad=10)
        
        # Add center crosshairs.
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        # Add grid lines.
        ax.grid(True, alpha=0.2)
        
        # Set ticks.
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        
        # Show the legend only in the first subplot.
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_trials, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Saccade 轨迹图 - Participant {participant_id}\n(绿色=起点, 红色=终点, 蓝色线=轨迹, 颜色=时间进度)', 
                 y=1.02, fontsize=14)
    plt.show()
    
    # 输出轨迹统计
    if verbose:
        print(f"\n📈 各 Trial 的 Saccade 轨迹统计:")
        for trial_num in trials:
            trial_saccades = valid_saccades[valid_saccades['trial_num'] == trial_num]
            if len(trial_saccades) > 0:
                # 计算轨迹总长度
                x_coords = trial_saccades['stimuli_x'].values
                y_coords = trial_saccades['stimuli_y'].values
                
                if len(x_coords) > 1:
                    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
                    total_distance = distances.sum()
                    max_distance = distances.max()
                    avg_distance = distances.mean()
                    
                    print(f"   Trial {int(trial_num)}: {len(trial_saccades)} 点, "
                          f"总轨迹长度: {total_distance:.3f}, "
                          f"最大步长: {max_distance:.3f}, "
                          f"平均步长: {avg_distance:.3f}")
                else:
                    print(f"   Trial {int(trial_num)}: {len(trial_saccades)} 点 (无轨迹)")


def visualize_saccade_trajectories_improved(data, participant_id, max_stimuli=None, 
                                           figsize_per_stimulus=(6, 6), verbose=True):
    """
    改进版saccade轨迹可视化 - 更直观易读
    
    Features:
    - 更大的点和线条
    - 数字标记显示时间顺序
    - 不同颜色区分不同时间段
    - 更清晰的起始结束标记
    - 简化的视觉元素
    
    Parameters:
    - data: 集成后的gaze数据 (DataFrame)
    - participant_id: 参与者ID (str)
    - max_stimuli: 最大显示的stimulus数量，None则显示所有 (int, optional)
    - figsize_per_stimulus: 每个子图的大小 (tuple)
    - verbose: 是否显示详细统计信息 (bool)
    
    Returns:
    - None (显示图表)
    """
    
    # 过滤saccade事件和有效坐标
    saccade_data = data[(data['event_type'] == 'saccade')].copy()
    valid_saccades = saccade_data.dropna(subset=['stimuli_x', 'stimuli_y', 'trial_num', 'device_time_stamp']).copy()
    
    if len(valid_saccades) == 0:
        print("❌ 没有有效的 saccade 轨迹数据")
        return
    
    # 按 trial_num 分组
    trials = sorted(valid_saccades['trial_num'].unique())
    if max_stimuli:
        trials = trials[:max_stimuli]
    
    if verbose:
        print(f"📊 创建 {len(trials)} 个 Stimulus 的改进版 Saccade 轨迹图")
    
    # 计算子图布局 - 使用更少的列数以便放大每个图
    n_trials = len(trials)
    n_cols = min(2, n_trials)  # 最多2列，让每个图更大
    n_rows = (n_trials + n_cols - 1) // n_cols
    
    # 创建更大的图形
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_stimulus[0] * n_cols, figsize_per_stimulus[1] * n_rows))
    
    # 处理单个子图的情况
    if n_trials == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 定义颜色方案 - 从冷色到暖色表示时间进度
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 蓝->紫->橙->红
    
    for idx, trial_num in enumerate(trials):
        row = idx // n_cols
        col = idx % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # 获取该trial的saccade数据
        trial_saccades = valid_saccades[valid_saccades['trial_num'] == trial_num].copy()
        
        if len(trial_saccades) == 0:
            ax.text(0.5, 0.5, 'No Saccade Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_title(f'Trial {int(trial_num)} - No Saccades', fontsize=14, pad=15)
            continue
        
        # 按时间排序
        trial_saccades = trial_saccades.sort_values('device_time_stamp').reset_index(drop=True)
        
        # 获取坐标
        x_coords = trial_saccades['stimuli_x'].values
        y_coords = trial_saccades['stimuli_y'].values
        n_points = len(x_coords)
        
        if n_points < 2:
            ax.scatter(x_coords[0], y_coords[0], c='blue', s=200, marker='o', 
                      label='Single Point', zorder=5, edgecolor='black', linewidth=2)
        else:
            # 将轨迹分成几个时间段，每段用不同颜色
            n_segments = min(4, max(2, n_points // 10))  # 2-4个段
            segment_size = n_points // n_segments
            
            for seg_idx in range(n_segments):
                start_idx = seg_idx * segment_size
                if seg_idx == n_segments - 1:  # 最后一段包含所有剩余点
                    end_idx = n_points
                else:
                    end_idx = (seg_idx + 1) * segment_size + 1  # +1 确保线条连续
                
                if end_idx > start_idx + 1:
                    # 绘制该段的轨迹线
                    color = colors[seg_idx % len(colors)]
                    ax.plot(x_coords[start_idx:end_idx], y_coords[start_idx:end_idx], 
                           color=color, linewidth=3, alpha=0.8, 
                           label=f'段 {seg_idx+1}' if seg_idx < 3 else None)
                    
                    # 在该段的点上添加标记
                    segment_points = range(start_idx, min(end_idx, n_points))
                    for i, point_idx in enumerate(segment_points):
                        if i % max(1, len(segment_points) // 5) == 0:  # 每段最多显示5个标记
                            ax.scatter(x_coords[point_idx], y_coords[point_idx], 
                                     c=color, s=80, alpha=0.7, zorder=4,
                                     edgecolor='white', linewidth=1)
            
            # 特殊标记起点和终点
            ax.scatter(x_coords[0], y_coords[0], c='lime', s=300, marker='*', 
                      label='起点', zorder=10, edgecolor='black', linewidth=2)
            ax.scatter(x_coords[-1], y_coords[-1], c='red', s=300, marker='X', 
                      label='终点', zorder=10, edgecolor='black', linewidth=2)
            
            # 添加数字标记显示关键点的顺序
            step = max(1, n_points // 8)  # 最多显示8个数字
            for i in range(0, n_points, step):
                if i < n_points - 1:  # 不在终点添加数字（因为已有X标记）
                    ax.annotate(f'{i+1}', (x_coords[i], y_coords[i]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=10, fontweight='bold', color='white',
                              bbox=dict(boxstyle='circle,pad=0.2', facecolor='black', alpha=0.7))
            
            # 添加一些关键统计信息
            total_distance = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
            ax.text(0.02, 0.98, f'总长度: {total_distance:.2f}\n点数: {n_points}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 设置坐标轴
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Stimuli X', fontsize=12)
        ax.set_ylabel('Stimuli Y', fontsize=12)
        
        # 获取trial信息
        condition = 'unknown'
        stimulus_id = 'unknown'
        if 'condition' in trial_saccades.columns:
            condition = trial_saccades['condition'].iloc[0] if len(trial_saccades) > 0 else 'unknown'
        if 'stimulus_id' in trial_saccades.columns:
            stimulus_id = trial_saccades['stimulus_id'].iloc[0] if len(trial_saccades) > 0 else 'unknown'
        
        # 创建清晰的标题
        title_text = f'Trial {int(trial_num)}'
        if stimulus_id != 'unknown':
            title_text += f' (Stimulus {stimulus_id})'
        subtitle = f'{condition}' if condition != 'unknown' else ''
        
        ax.set_title(f'{title_text}\n{subtitle}', fontsize=14, pad=15, fontweight='bold')
        
        # 添加更明显的中心线和象限标记
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=2)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=2)
        
        # 添加象限标签
        ax.text(0.5, 0.5, '右上', transform=ax.transAxes, ha='center', va='center', 
               alpha=0.3, fontsize=12, fontweight='bold')
        ax.text(-0.5, 0.5, '左上', transform=ax.transData, ha='center', va='center', 
               alpha=0.3, fontsize=12, fontweight='bold')
        ax.text(-0.5, -0.5, '左下', transform=ax.transData, ha='center', va='center', 
               alpha=0.3, fontsize=12, fontweight='bold')
        ax.text(0.5, -0.5, '右下', transform=ax.transData, ha='center', va='center', 
               alpha=0.3, fontsize=12, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        
        # 设置更清晰的刻度
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(labelsize=10)
        
        # 只在第一个子图显示图例
        if idx == 0 and n_points >= 2:
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 隐藏多余的子图
    for idx in range(n_trials, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Saccade 轨迹分析 - Participant {participant_id}\n' + 
                 '⭐起点 ❌终点 | 颜色=时间进度 | 数字=顺序', 
                 y=0.98, fontsize=16, fontweight='bold')
    plt.show()
    
    # 输出更详细的轨迹统计
    if verbose:
        print(f"\n📈 详细轨迹统计:")
        for trial_num in trials:
            trial_saccades = valid_saccades[valid_saccades['trial_num'] == trial_num]
            if len(trial_saccades) > 0:
                x_coords = trial_saccades['stimuli_x'].values
                y_coords = trial_saccades['stimuli_y'].values
                
                if len(x_coords) > 1:
                    distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
                    total_distance = distances.sum()
                    
                    # 计算轨迹的"直线性"(起点到终点的直线距离 vs 实际轨迹长度)
                    straight_distance = np.sqrt((x_coords[-1] - x_coords[0])**2 + 
                                              (y_coords[-1] - y_coords[0])**2)
                    linearity = straight_distance / total_distance if total_distance > 0 else 0
                    
                    # 计算轨迹的"分散度"(标准差)
                    x_spread = np.std(x_coords)
                    y_spread = np.std(y_coords)
                    
                    print(f"   Trial {int(trial_num)}: {len(trial_saccades)} 点")
                    print(f"     总轨迹长度: {total_distance:.3f}")
                    print(f"     直线性指数: {linearity:.3f} (1=完全直线, 0=非常曲折)")
                    print(f"     空间分散度: X={x_spread:.3f}, Y={y_spread:.3f}")
                    print(f"     平均步长: {distances.mean():.3f}")
                else:
                    print(f"   Trial {int(trial_num)}: {len(trial_saccades)} 点 (单点，无轨迹)")


def load_and_visualize_participant(participant_id, data_file_path, visualization_type='heatmap',
                                  max_trials=None, show_aoi_boundaries=True, figsize_per_trial=(4, 4),
                                  style='academic', verbose=True, use_original_aoi=True):
    """
    便捷函数：加载参与者数据并进行可视化

    Parameters:
    - participant_id: 参与者ID (str)
    - data_file_path: 数据文件路径 (str)
    - visualization_type: 可视化类型 ('heatmap', 'saccade_trajectory', 'saccade_improved') (str)
    - max_trials: 最大显示的trial数量，None则显示所有 (int, optional)
    - show_aoi_boundaries: 是否显示AOI边界线（仅对heatmap有效） (bool)
    - figsize_per_trial: 每个子图的大小 (tuple)
    - style: 可视化风格 ('academic'=学术白色背景, 'modern'=现代深色背景) (str)
    - verbose: 是否显示详细统计信息 (bool)
    - use_original_aoi: True=使用原始5个AOI, False=使用合并的3个AOI (bool)

    Returns:
    - DataFrame: 加载的数据
    """
    
    try:
        # 加载数据
        data = pd.read_csv(data_file_path)
        if verbose:
            print(f"✅ 成功加载参与者 {participant_id} 的数据: {len(data)} 行")
            
            # 显示数据概览
            if 'event_type' in data.columns:
                event_counts = data['event_type'].value_counts()
                total = len(data)
                print(f"📊 Event Type 分布:")
                for event, count in event_counts.items():
                    pct = count / total * 100
                    print(f"   {event}: {count:,} 点 ({pct:.1f}%)")
        
        # 根据类型进行可视化
        if visualization_type == 'heatmap':
            visualize_gaze_heatmap_by_trial(
                data=data,
                participant_id=participant_id,
                max_trials=max_trials,
                figsize_per_trial=figsize_per_trial,
                show_aoi_boundaries=show_aoi_boundaries,
                style=style,
                verbose=verbose,
                use_original_aoi=use_original_aoi
            )
        elif visualization_type == 'saccade_trajectory':
            visualize_saccade_trajectories_by_stimulus(
                data=data,
                participant_id=participant_id,
                max_stimuli=max_trials,
                figsize_per_stimulus=figsize_per_trial,
                verbose=verbose
            )
        elif visualization_type == 'saccade_improved':
            visualize_saccade_trajectories_improved(
                data=data,
                participant_id=participant_id,
                max_stimuli=max_trials,
                figsize_per_stimulus=figsize_per_trial,
                verbose=verbose
            )
        else:
            print(f"❌ 未知的可视化类型: {visualization_type}")
            print("   支持的类型: 'heatmap', 'saccade_trajectory', 'saccade_improved'")
            return data
        
        return data
        
    except FileNotFoundError:
        print(f"❌ 未找到数据文件: {data_file_path}")
        return None
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        return None


# 便捷使用函数
def quick_heatmap(participant_id, data_file_path, max_trials=None, show_aoi=True, style='academic', use_original_aoi=True):
    """快速生成热力图 - 支持学术和现代两种风格

    Parameters:
    - participant_id: 参与者ID (str)
    - data_file_path: 数据文件路径 (str)
    - max_trials: 最大显示的trial数量 (int, optional)
    - show_aoi: 是否显示AOI边界 (bool)
    - style: 可视化风格 ('academic' or 'modern') (str)
    - use_original_aoi: True=使用原始5个AOI, False=使用合并的3个AOI (bool)
    """
    return load_and_visualize_participant(
        participant_id=participant_id,
        data_file_path=data_file_path,
        visualization_type='heatmap',
        max_trials=max_trials,
        show_aoi_boundaries=show_aoi,
        style=style,
        verbose=True,
        use_original_aoi=use_original_aoi
    )


def quick_saccade_trajectory(participant_id, data_file_path, max_trials=None):
    """快速生成saccade轨迹图"""
    return load_and_visualize_participant(
        participant_id=participant_id,
        data_file_path=data_file_path,
        visualization_type='saccade_trajectory',
        max_trials=max_trials,
        verbose=True
    )


def quick_saccade_improved(participant_id, data_file_path, max_trials=None):
    """快速生成改进版saccade轨迹图"""
    return load_and_visualize_participant(
        participant_id=participant_id,
        data_file_path=data_file_path,
        visualization_type='saccade_improved',
        max_trials=max_trials,
        verbose=True
    )

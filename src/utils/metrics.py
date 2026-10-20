"""
Aggregate the per-subject evaluation metrics (the cv_metrics_*.json files written by
evaluate.py) into LaTeX tables (mean and standard deviation over subjects) and a figure
of the median metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from matplotlib.patches import Patch


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
    pass


class DataProcessingError(Exception):
    """Raised when there's an error processing data."""
    pass


def get_latex_template_psnr_ssim() -> str:
    """Return the LaTeX template for PSNR and SSIM metrics (combined format)."""
    return r"""\begin{{table*}}[htbp]
\caption{{PSNR and SSIM metrics across training time.}}
\renewcommand{{\arraystretch}}{{1.2}}
\centering
\begin{{tabular}}{{c c c l l l}}
\toprule
\textbf{{Metric}} & \textbf{{Map}} & \textbf{{Method}} & \textbf{{5 min}} & \textbf{{10 min}} & \textbf{{30 min}} \\
\midrule
\multirow{{6}}{{*}}{{\makecell{{\textbf{{PSNR (dB)}} $\uparrow$}}}}
& \multirow{{3}}{{*}}{{FA}}
& Baseline & {PSNR_FA_baseline_5} & {PSNR_FA_baseline_10} & {PSNR_FA_baseline_30} \\
& & HashEnc & {PSNR_FA_hashenc_5} & {PSNR_FA_hashenc_10} & {PSNR_FA_hashenc_30} \\
& & Proposed & {PSNR_FA_proposed_5} & {PSNR_FA_proposed_10} & {PSNR_FA_proposed_30} \\
\cmidrule(lr){{2-6}}
& \multirow{{3}}{{*}}{{GFA}}
& Baseline & {PSNR_GFA_baseline_5} & {PSNR_GFA_baseline_10} & {PSNR_GFA_baseline_30} \\
& & HashEnc & {PSNR_GFA_hashenc_5} & {PSNR_GFA_hashenc_10} & {PSNR_GFA_hashenc_30} \\
& & Proposed & {PSNR_GFA_proposed_5} & {PSNR_GFA_proposed_10} & {PSNR_GFA_proposed_30} \\
\midrule
\multirow{{6}}{{*}}{{\makecell{{\textbf{{SSIM}} $\uparrow$}}}}
& \multirow{{3}}{{*}}{{FA}}
& Baseline & {SSIM_FA_baseline_5} & {SSIM_FA_baseline_10} & {SSIM_FA_baseline_30} \\
& & HashEnc & {SSIM_FA_hashenc_5} & {SSIM_FA_hashenc_10} & {SSIM_FA_hashenc_30} \\
& & Proposed & {SSIM_FA_proposed_5} & {SSIM_FA_proposed_10} & {SSIM_FA_proposed_30} \\
\cmidrule(lr){{2-6}}
& \multirow{{3}}{{*}}{{GFA}}
& Baseline & {SSIM_GFA_baseline_5} & {SSIM_GFA_baseline_10} & {SSIM_GFA_baseline_30} \\
& & HashEnc & {SSIM_GFA_hashenc_5} & {SSIM_GFA_hashenc_10} & {SSIM_GFA_hashenc_30} \\
& & Proposed & {SSIM_GFA_proposed_5} & {SSIM_GFA_proposed_10} & {SSIM_GFA_proposed_30} \\
\bottomrule
\end{{tabular}}
\label{{tab:cv-metrics-psnr-ssim}}
\end{{table*}}
"""


def get_latex_template_nrmse_fsim() -> str:
    """Return the LaTeX template for NRMSE and FSIM metrics (combined format)."""
    return r"""\begin{{table*}}[htbp]
\caption{{NRMSE and FSIM metrics across training time.}}
\renewcommand{{\arraystretch}}{{1.2}}
\centering
\begin{{tabular}}{{c c c l l l}}
\toprule
\textbf{{Metric}} & \textbf{{Map}} & \textbf{{Method}} & \textbf{{5 min}} & \textbf{{10 min}} & \textbf{{30 min}} \\
\midrule
\multirow{{6}}{{*}}{{\makecell{{\textbf{{NRMSE}} $\downarrow$}}}}
& \multirow{{3}}{{*}}{{FA}}
& Baseline & {NRMSE_FA_baseline_5} & {NRMSE_FA_baseline_10} & {NRMSE_FA_baseline_30} \\
& & HashEnc & {NRMSE_FA_hashenc_5} & {NRMSE_FA_hashenc_10} & {NRMSE_FA_hashenc_30} \\
& & Proposed & {NRMSE_FA_proposed_5} & {NRMSE_FA_proposed_10} & {NRMSE_FA_proposed_30} \\
\cmidrule(lr){{2-6}}
& \multirow{{3}}{{*}}{{GFA}}
& Baseline & {NRMSE_GFA_baseline_5} & {NRMSE_GFA_baseline_10} & {NRMSE_GFA_baseline_30} \\
& & HashEnc & {NRMSE_GFA_hashenc_5} & {NRMSE_GFA_hashenc_10} & {NRMSE_GFA_hashenc_30} \\
& & Proposed & {NRMSE_GFA_proposed_5} & {NRMSE_GFA_proposed_10} & {NRMSE_GFA_proposed_30} \\
\midrule
\multirow{{6}}{{*}}{{\makecell{{\textbf{{FSIM}} $\uparrow$}}}}
& \multirow{{3}}{{*}}{{FA}}
& Baseline & {FSIM_FA_baseline_5} & {FSIM_FA_baseline_10} & {FSIM_FA_baseline_30} \\
& & HashEnc & {FSIM_FA_hashenc_5} & {FSIM_FA_hashenc_10} & {FSIM_FA_hashenc_30} \\
& & Proposed & {FSIM_FA_proposed_5} & {FSIM_FA_proposed_10} & {FSIM_FA_proposed_30} \\
\cmidrule(lr){{2-6}}
& \multirow{{3}}{{*}}{{GFA}}
& Baseline & {FSIM_GFA_baseline_5} & {FSIM_GFA_baseline_10} & {FSIM_GFA_baseline_30} \\
& & HashEnc & {FSIM_GFA_hashenc_5} & {FSIM_GFA_hashenc_10} & {FSIM_GFA_hashenc_30} \\
& & Proposed & {FSIM_GFA_proposed_5} & {FSIM_GFA_proposed_10} & {FSIM_GFA_proposed_30} \\
\bottomrule
\end{{tabular}}
\label{{tab:cv-metrics-nrmse-fsim}}
\end{{table*}}
"""


def resolve_path_pattern(pattern: str, method: str, time: str, index: int) -> Path:
    """
    Resolve a path pattern (potentially with wildcards) to a single file.

    Args:
        pattern: Path pattern (can contain wildcards like *)
        method: Method name (for error reporting)
        time: Time period (for error reporting)
        index: Index in the list (for error reporting)

    Returns:
        Resolved Path object

    Raises:
        ConfigurationError: If pattern matches 0 or >1 files
    """
    # Check if pattern contains wildcards
    if '*' in pattern or '?' in pattern or '[' in pattern:
        matches = glob.glob(pattern)

        if len(matches) == 0:
            raise ConfigurationError(
                f"No files match pattern '{pattern}' "
                f"(method={method}, time={time}, index={index})"
            )
        elif len(matches) > 1:
            files_list = '\n  - '.join(matches)
            raise ConfigurationError(
                f"Multiple files match pattern '{pattern}' "
                f"(method={method}, time={time}, index={index}):\n  - {files_list}"
            )

        # Exactly one match - resolve it
        return Path(matches[0]).resolve()
    else:
        # No wildcards - just resolve the path
        return Path(pattern).resolve()


def load_paths_config(config_path: str) -> Dict:
    """Load the JSON configuration file containing paths."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")


def load_json_files(json_paths: List[Path]) -> List[Dict]:
    """Load all JSON files and return list of dictionaries."""
    data = []
    for path in json_paths:
        try:
            with open(path, 'r') as f:
                data.append(json.load(f))
        except Exception as e:
            warnings.warn(f"Could not load {path}: {e}")
    return data


def compute_statistics_for_group(json_paths: List[Path]) -> Dict[str, Dict[str, tuple]]:
    """
    Compute mean and std for each metric from a list of JSON files.

    Args:
        json_paths: List of resolved Path objects

    Returns:
        Dictionary with structure: {metric: {category: (mean, std)}}
    """
    # Load all JSON files
    data = load_json_files(json_paths)

    if not data:
        warnings.warn("No data loaded from paths")
        return {}

    # Metrics and categories to extract
    metrics = ['PSNR', 'SSIM', 'NRMSE', 'FSIM']
    categories = ['FA', 'GFA']

    # Collect values for each metric and category
    stats = {}
    for metric in metrics:
        stats[metric] = {}
        for category in categories:
            values = []
            for entry in data:
                if metric in entry and category in entry[metric]:
                    values.append(entry[metric][category])

            if values:
                mean = np.mean(values)
                std = np.std(values)
                stats[metric][category] = (mean, std)
            else:
                stats[metric][category] = (0.0, 0.0)
                warnings.warn(f"No data found for {metric}/{category}")

    return stats


def find_best_values(all_stats: Dict, metrics_list: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Find the best method for each metric/category/time combination.

    Args:
        all_stats: Dictionary with all statistics
        metrics_list: List of metrics to consider

    Returns:
        Dictionary mapping metric -> category -> time -> best_method
    """
    methods = ['baseline', 'hashenc', 'proposed']
    times = ['5_min', '10_min', '30_min']
    categories = ['FA', 'GFA']

    # Metrics that are "higher is better"
    higher_is_better = {'PSNR': True, 'SSIM': True, 'NRMSE': False, 'FSIM': True}

    best_values = {}

    for metric in metrics_list:
        best_values[metric] = {}
        for category in categories:
            best_values[metric][category] = {}
            for time in times:
                # Collect all values for this combination
                values = {}
                for method in methods:
                    if (method in all_stats and
                        time in all_stats[method] and
                        metric in all_stats[method][time] and
                        category in all_stats[method][time][metric]):
                        mean, _ = all_stats[method][time][metric][category]
                        values[method] = mean

                # Find best method
                if values:
                    if higher_is_better[metric]:
                        best_method = max(values, key=values.get)
                    else:
                        best_method = min(values, key=values.get)
                    best_values[metric][category][time] = best_method
                else:
                    best_values[metric][category][time] = None

    return best_values


def format_combined_value(mean: float, std: float, baseline_mean: float,
                         is_baseline: bool, is_higher_better: bool, is_bold: bool = False,
                         precision: int = 3, std_precision: int = 4, pct_precision: int = 2) -> str:
    """
    Format mean ± std (pct%) as LaTeX string.

    Args:
        mean: Mean value
        std: Standard deviation
        baseline_mean: Baseline mean for percentage calculation
        is_baseline: Whether this is the baseline row
        is_higher_better: True if higher values are better
        is_bold: Whether to bold the value
        precision: Number of decimal places for mean
        std_precision: Number of decimal places for std
        pct_precision: Number of decimal places for percentage

    Returns:
        LaTeX formatted string
    """
    mean_str = f"{mean:.{precision}f}"
    std_str = f"{std:.{std_precision}f}"

    if is_bold:
        mean_str = f"\\textbf{{{mean_str}}}"

    # Format: mean ±std
    base_str = f"{mean_str} {{\\scriptsize \\textcolor{{gray}}{{$\\pm${std_str}}}}}"

    # If baseline, no percentage
    if is_baseline:
        return base_str

    # Calculate percentage improvement
    if baseline_mean == 0:
        return base_str  # Avoid division by zero

    pct_change = ((mean - baseline_mean) / abs(baseline_mean)) * 100

    # Determine if this is an improvement based on metric type
    if is_higher_better:
        is_improvement = pct_change > 0
    else:
        is_improvement = pct_change < 0
        pct_change = -pct_change  # Flip sign for display (so reduction is positive)

    # Format percentage
    pct_str = f"{abs(pct_change):.{pct_precision}f}"

    # Add color and sign
    if is_improvement:
        color_pct = f"{{\\textcolor{{green!70!black}}{{(+{pct_str}\\%)}}}}"
    else:
        color_pct = f"{{\\textcolor{{red}}{{(-{pct_str}\\%)}}}}"

    return f"{base_str} {color_pct}"


def validate_and_resolve_config(config: Dict, dataset: str) -> Dict[str, Dict[str, List[Path]]]:
    """
    Validate configuration and resolve all path patterns.

    Args:
        config: Configuration dictionary with potentially unresolved paths
        dataset: Dataset name to use from the configuration

    Returns:
        Configuration with resolved Path objects

    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check if dataset exists
    if dataset not in config:
        raise ConfigurationError(f"Dataset '{dataset}' not found in configuration.")

    dataset_config = config[dataset]

    required_methods = ['baseline', 'hashenc', 'proposed']
    required_times = ['5_min', '10_min', '30_min']

    resolved_config = {}

    # Check all methods exist
    for method in required_methods:
        if method not in dataset_config:
            raise ConfigurationError(
                f"Method '{method}' not found in dataset '{dataset}'"
            )

        resolved_config[method] = {}

        # Check all time periods exist
        for time in required_times:
            if time not in dataset_config[method]:
                raise ConfigurationError(
                    f"Time '{time}' not found for method '{method}' in dataset '{dataset}'"
                )

            # Check that it's a list
            if not isinstance(dataset_config[method][time], list):
                raise ConfigurationError(
                    f"'{dataset}.{method}.{time}' must be a list of paths"
                )

            # Resolve all paths/patterns in this list
            resolved_paths = []
            for idx, pattern in enumerate(dataset_config[method][time]):
                resolved_path = resolve_path_pattern(pattern, method, time, idx)

                # Check that file exists
                if not resolved_path.exists():
                    raise ConfigurationError(
                        f"File does not exist: {resolved_path} "
                        f"(dataset={dataset}, method={method}, time={time}, index={idx})"
                    )

                resolved_paths.append(resolved_path)

            resolved_config[method][time] = resolved_paths

    # Check that all time periods have the same number of files
    num_files = None
    for method in required_methods:
        for time in required_times:
            count = len(resolved_config[method][time])
            if num_files is None:
                num_files = count
            elif count != num_files:
                raise ConfigurationError(
                    f"All methods and times must have the same number of files. "
                    f"Expected {num_files}, got {count} for {dataset}.{method}.{time}"
                )

    return resolved_config


def save_resolved_config(resolved_config: Dict[str, Dict[str, List[Path]]],
                         dataset: str,
                         output_path: str) -> None:
    """
    Save resolved configuration to JSON file.

    Args:
        resolved_config: Configuration with resolved Path objects
        dataset: Dataset name
        output_path: Path to save the resolved configuration
    """
    # Convert Path objects to strings
    output_config = {dataset: {}}

    for method, times in resolved_config.items():
        output_config[dataset][method] = {}
        for time, paths in times.items():
            output_config[dataset][method][time] = [str(p) for p in paths]

    # Write to file
    output_path_obj = Path(output_path).resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, 'w') as f:
        json.dump(output_config, f, indent=4)

    print(f"Resolved paths saved to: {output_path_obj}")


def collect_median_metrics_data(resolved_config: Dict[str, Dict[str, List[Path]]]) -> pd.DataFrame:
    """
    Collect data for the three median metrics across all methods and times.

    Args:
        resolved_config: Configuration with resolved Path objects

    Returns:
        DataFrame with columns: Method, Time, Metric, Value
    """
    methods = ['baseline', 'hashenc', 'proposed']
    method_labels = {'baseline': 'Baseline', 'hashenc': 'HashEnc', 'proposed': 'Proposed'}
    times = ['5_min', '10_min', '30_min']
    time_labels = {'5_min': '5 min', '10_min': '10 min', '30_min': '30 min'}

    # Metrics to extract
    median_metrics = [
        'ODF_L2_Norm_Error_median',
        'GFA_Absolute_Error_median',
        'Eigenvector_Angular_Error_median_deg'
    ]

    # Better metric names for display
    metric_display_names = {
        'ODF_L2_Norm_Error_median': 'ODF L2 Norm Error',
        'GFA_Absolute_Error_median': 'GFA Absolute Error',
        'Eigenvector_Angular_Error_median_deg': 'Eigenvector Angular Error (°)'
    }

    data_rows = []

    for method in methods:
        for time in times:
            json_paths = resolved_config[method][time]

            # Load all JSON files for this method/time
            json_data = load_json_files(json_paths)

            # Extract each median metric
            for metric in median_metrics:
                for entry in json_data:
                    if metric in entry:
                        value = entry[metric]
                        data_rows.append({
                            'Method': method_labels[method],
                            'Time': time_labels[time],
                            'Metric': metric_display_names[metric],
                            'Value': value
                        })

    return pd.DataFrame(data_rows)


def create_median_metrics_plot(df: pd.DataFrame,
                               output_path: Path,
                               plot_type: str = 'boxen',
                               # Figure dimensions
                               figure_width: float = 16,
                               figure_height: float = 5,
                               figure_dpi: int = 300,
                               # Font sizes
                               base_font_size: int = 22,
                               title_font_size: int = 22,
                               axis_label_font_size: int = 22,
                               tick_label_font_size: int = 22,
                               legend_font_size: int = 20,
                               # Plot element sizes
                               box_line_width: float = 1.2,
                               box_width: float = 0.7,
                               box_gap: float = 0.15,
                               flier_size: float = 3,
                               median_line_width: float = 1.5,
                               # Violin plot specific
                               violin_line_width: float = 1.0,
                               violin_bandwidth_adjust: float = 0.5,
                               violin_width: float = 0.35,
                               inner_box_width: float = 0.12,
                               inner_box_line_width: float = 1.5,
                               # Grid and separator styling
                               grid_line_width: float = 0.8,
                               grid_alpha: float = 0.6,
                               separator_line_width: float = 1.8,
                               separator_alpha: float = 0.7,
                               spine_line_width: float = 1.0,
                               # Spacing (for boxen/violin types)
                               method_spacing: float = 0.5,
                               separator_space: float = 0.6,
                               # Padding
                               xlabel_pad: float = 12,
                               ylabel_pad: float = 12,
                               title_pad: float = 18,
                               tick_pad: float = 8) -> None:
    """
    Create box/violin plots for median metrics with publication-ready formatting.

    Args:
        df: DataFrame with columns Method, Time, Metric, Value
        output_path: Path to save the plot
        plot_type: 'box', 'violin', or 'boxen' (violin with box inside)

        Figure dimensions:
        figure_width: Figure width in inches (default: 16)
        figure_height: Figure height in inches (default: 5)
        figure_dpi: DPI for saved figure (default: 300 for publication quality)

        Font sizes:
        base_font_size: Base font size for general text (default: 22)
        title_font_size: Subplot title font size (default: 22)
        axis_label_font_size: Axis label font size (default: 22)
        tick_label_font_size: Tick label font size (default: 22)
        legend_font_size: Legend font size (default: 20)

        Plot element sizes:
        box_line_width: Line width for box plot edges (default: 1.2)
        box_width: Width of boxes (default: 0.7)
        box_gap: Gap between boxes (default: 0.15)
        flier_size: Size of outlier markers (default: 3)
        median_line_width: Width of median line (default: 1.5)

        Violin plot specific:
        violin_line_width: Line width for violin edges (default: 1.0)
        violin_bandwidth_adjust: Bandwidth adjustment for violin smoothness (default: 0.5)
        violin_width: Width of violin plots in boxen mode (default: 0.35)
        inner_box_width: Width of inner box in boxen mode (default: 0.12)
        inner_box_line_width: Line width for inner box in boxen mode (default: 1.5)

        Grid and separator styling:
        grid_line_width: Width of grid lines (default: 0.8)
        grid_alpha: Transparency of grid lines (default: 0.6)
        separator_line_width: Width of separator lines between time groups (default: 1.8)
        separator_alpha: Transparency of separator lines (default: 0.7)
        spine_line_width: Width of axis spines (default: 1.0)

        Spacing:
        method_spacing: Spacing between methods in boxen mode (default: 0.5)
        separator_space: Space allocated for separators (default: 0.6)

        Padding:
        xlabel_pad: Padding for x-axis label (default: 12)
        ylabel_pad: Padding for y-axis label (default: 12)
        title_pad: Padding for subplot titles (default: 18)
        tick_pad: Padding for tick labels (default: 8)
    """
    # Set style
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", rc={
        "font.size": base_font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": axis_label_font_size,
        "xtick.labelsize": tick_label_font_size,
        "ytick.labelsize": tick_label_font_size,
        "legend.fontsize": legend_font_size,
    })

    # Solid colors - no transparency
    method_colors = {
        'Baseline': '#EA8E83',    # Coral/red
        'HashEnc': '#8AAFC9',     # Light blue
        'Proposed': '#AFCF78'     # Lime green
    }

    # Get unique metrics
    metrics = df['Metric'].unique()
    n_metrics = len(metrics)

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(figure_width, figure_height), sharey=False, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.1, wspace=0.05)

    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric in its own subplot
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        # Filter data for this metric
        metric_df = df[df['Metric'] == metric]

        # Create the plot based on type
        if plot_type == 'box':
            sns.boxplot(
                data=metric_df,
                x='Time',
                y='Value',
                hue='Method',
                palette=method_colors,
                ax=ax,
                linewidth=box_line_width,
                fliersize=flier_size,
                width=box_width,
                gap=box_gap,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=flier_size,
                               linestyle='none', markeredgewidth=0, alpha=0.5),
                medianprops=dict(linewidth=median_line_width, color='white', solid_capstyle='butt'),
                whiskerprops=dict(linewidth=box_line_width),
                capprops=dict(linewidth=box_line_width),
                saturation=1.0
            )
            # Set equal x-axis limits for all time groups
            ax.set_xlim(-0.5, 2.5)

        elif plot_type == 'violin':
            sns.violinplot(
                data=metric_df,
                x='Time',
                y='Value',
                hue='Method',
                palette=method_colors,
                ax=ax,
                inner='quartile',
                linewidth=violin_line_width,
                cut=1,
                gap=box_gap,
                bw_adjust=violin_bandwidth_adjust,
                saturation=1.0
            )
            # Set equal x-axis limits for all time groups
            ax.set_xlim(-0.5, 2.5)

        elif plot_type == 'boxen':
            # Use matplotlib's violinplot for proper positioning
            times = ['5 min', '10 min', '30 min']
            methods = ['Baseline', 'HashEnc', 'Proposed']

            # Calculate positions with equal spacing
            group_width = 3 * method_spacing  # Width for 3 methods

            # Positions for each time group (centered)
            time_centers = [1.5, 4.8, 8.1]

            plot_data = []
            positions = []
            colors_list = []

            for time_idx, time in enumerate(times):
                center = time_centers[time_idx]
                method_positions = [center - method_spacing, center, center + method_spacing]

                for method_idx, method in enumerate(methods):
                    data = metric_df[(metric_df['Time'] == time) & (metric_df['Method'] == method)]['Value'].values
                    if len(data) > 0:
                        plot_data.append(data)
                        positions.append(method_positions[method_idx])
                        colors_list.append(method_colors[method])

            # Create violin plots with matplotlib
            parts = ax.violinplot(
                plot_data,
                positions=positions,
                widths=violin_width,
                showmeans=False,
                showextrema=False,
                showmedians=False
            )

            # Color the violin plots - solid, no transparency
            for pc, color in zip(parts['bodies'], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(1.0)  # Solid
                pc.set_edgecolor('none')
                pc.set_zorder(2)

            # Overlay thin box plots (dark gray)
            bp = ax.boxplot(
                plot_data,
                positions=positions,
                widths=inner_box_width,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='none', markersize=flier_size * 0.83,
                               linestyle='none', markeredgewidth=0.8, alpha=0.6, zorder=4),
                boxprops=dict(linewidth=inner_box_line_width, edgecolor='#333333', zorder=3),
                whiskerprops=dict(linewidth=box_line_width, color='#333333', zorder=3),
                capprops=dict(linewidth=box_line_width, color='#333333', zorder=3),
                medianprops=dict(linewidth=box_line_width, color='white', solid_capstyle='butt', zorder=5)
            )

            # Color boxes with dark gray (solid)
            for patch in bp['boxes']:
                patch.set_facecolor('#555555')
                patch.set_alpha(1.0)  # Solid

            # Set x-axis labels and limits with equal spacing
            ax.set_xticks(time_centers)
            ax.set_xticklabels(times)
            ax.set_xlim(0.5, 9.6)  # Equal padding on both sides

            # Add vertical separator lines between time groups (centered between groups)
            separator_positions = [(time_centers[0] + time_centers[1]) / 2,
                                  (time_centers[1] + time_centers[2]) / 2]
            for sep_x in separator_positions:
                ax.axvline(x=sep_x, color='#555555', linestyle='--',
                          linewidth=separator_line_width, alpha=separator_alpha, zorder=1)

            # Create custom legend with colored patches
            legend_elements = [Patch(facecolor=method_colors[m], edgecolor='#333333',
                                    linewidth=box_line_width, label=m, alpha=1.0)
                             for m in methods]
            ax.legend(handles=legend_elements, loc='upper right',
                    bbox_to_anchor=(1.0, 1.0), borderpad=0, borderaxespad=0,
                    frameon=False, fontsize=legend_font_size, labelspacing=0.2)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        # Add vertical separators for box and violin plots
        if plot_type in ['box', 'violin']:
            for i in range(len(['5 min', '10 min', '30 min']) - 1):
                sep_pos = i + 0.5
                ax.axvline(x=sep_pos, color='#555555', linestyle='--',
                          linewidth=separator_line_width, alpha=separator_alpha, zorder=1)

        # Customize subplot
        ax.set_xlabel('Training Time', fontsize=axis_label_font_size, fontweight='bold', labelpad=xlabel_pad)
        ax.set_title(metric, fontsize=title_font_size, fontweight='bold', pad=title_pad)

        # Improve tick labels with padding
        ax.tick_params(axis='both', which='major', labelsize=tick_label_font_size, pad=tick_pad)
        ax.tick_params(axis='x', rotation=0)

        # Grid styling - only horizontal lines, more visible
        ax.grid(axis='y', alpha=grid_alpha, linestyle='-', linewidth=grid_line_width, color='#CCCCCC', zorder=0)
        ax.grid(axis='x', visible=False)
        ax.set_axisbelow(True)

        # Clean background
        ax.set_facecolor('white')

        # Spine styling - remove top and right
        sns.despine(ax=ax, left=False, bottom=False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(spine_line_width)
            ax.spines[spine].set_color('#555555')

        # Legend in upper right corner, no frame (for box and violin types)
        if plot_type != 'boxen':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles,
                labels=labels,
                loc='upper right',
                frameon=False,
                fontsize=legend_font_size
            )

    # Clean figure background
    fig.patch.set_facecolor('white')

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=figure_dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")

    plt.close()


def generate_latex_tables(config: Dict,
                          dataset: str,
                          output_dir: str,
                          resolved_path: str,
                          precision: int = 3,
                          std_precision: int = 4,
                          pct_precision: int = 2) -> Dict[str, Dict[str, List[Path]]]:
    """
    Generate LaTeX tables from configuration and return the resolved configuration.

    Args:
        config: Configuration dictionary with path patterns
        dataset: Dataset name to use
        output_dir: Directory to save the .tex files
        resolved_path: Path to save the resolved configuration
        precision: Number of decimal places for mean values
        std_precision: Number of decimal places for std
        pct_precision: Number of decimal places for percentages
    """
    # Validate and resolve all paths
    print(f"Using dataset: {dataset}")
    print("Validating and resolving paths...")
    resolved_config = validate_and_resolve_config(config, dataset)
    print("All paths validated and resolved successfully!\n")

    # Save resolved configuration
    save_resolved_config(resolved_config, dataset, resolved_path)
    print()

    # Define structure
    methods = ['baseline', 'hashenc', 'proposed']
    times = ['5_min', '10_min', '30_min']
    time_labels = {'5_min': '5', '10_min': '10', '30_min': '30'}
    categories = ['FA', 'GFA']
    metrics = ['PSNR', 'SSIM', 'NRMSE', 'FSIM']

    # Metrics that are "higher is better"
    higher_is_better = {'PSNR': True, 'SSIM': True, 'NRMSE': False, 'FSIM': True}

    # Store all statistics for percentage calculations
    all_stats = {}

    # Process each method and time period
    for method in methods:
        all_stats[method] = {}
        for time in times:
            time_label = time_labels[time]
            json_paths = resolved_config[method][time]

            print(f"Processing {method} - {time}: {len(json_paths)} files")

            # Compute statistics for this group
            stats = compute_statistics_for_group(json_paths)
            all_stats[method][time] = stats

    print("\nGenerating LaTeX tables...\n")

    # Create output directory
    output_path_obj = Path(output_dir).resolve()
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Find best values for each metric group
    best_values_psnr_ssim = find_best_values(all_stats, ['PSNR', 'SSIM'])
    best_values_nrmse_fsim = find_best_values(all_stats, ['NRMSE', 'FSIM'])

    # Table 1: PSNR/SSIM combined format
    substitutions_1 = {}
    for metric in ['PSNR', 'SSIM']:
        for category in categories:
            for method in methods:
                for time in times:
                    time_label = time_labels[time]
                    stats = all_stats[method][time]
                    baseline_stats = all_stats['baseline'][time]

                    if metric in stats and category in stats[metric]:
                        mean, std = stats[metric][category]
                        baseline_mean, _ = baseline_stats[metric][category]
                        is_bold = (best_values_psnr_ssim[metric][category][time] == method)
                        is_baseline = (method == 'baseline')

                        formatted = format_combined_value(
                            mean, std, baseline_mean, is_baseline,
                            higher_is_better[metric], is_bold,
                            precision, std_precision, pct_precision
                        )

                        key = f"{metric}_{category}_{method}_{time_label}"
                        substitutions_1[key] = formatted
                    else:
                        key = f"{metric}_{category}_{method}_{time_label}"
                        substitutions_1[key] = "N/A"

    latex_output_1 = get_latex_template_psnr_ssim().format(**substitutions_1)
    output_file_1 = output_path_obj / "cv_metrics_psnr_ssim_combined.tex"
    with open(output_file_1, 'w') as f:
        f.write(latex_output_1)
    print(f"Table 1 (PSNR/SSIM combined) written to: {output_file_1}")

    # Table 2: NRMSE/FSIM combined format
    substitutions_2 = {}
    for metric in ['NRMSE', 'FSIM']:
        for category in categories:
            for method in methods:
                for time in times:
                    time_label = time_labels[time]
                    stats = all_stats[method][time]
                    baseline_stats = all_stats['baseline'][time]

                    if metric in stats and category in stats[metric]:
                        mean, std = stats[metric][category]
                        baseline_mean, _ = baseline_stats[metric][category]
                        is_bold = (best_values_nrmse_fsim[metric][category][time] == method)
                        is_baseline = (method == 'baseline')

                        formatted = format_combined_value(
                            mean, std, baseline_mean, is_baseline,
                            higher_is_better[metric], is_bold,
                            precision, std_precision, pct_precision
                        )

                        key = f"{metric}_{category}_{method}_{time_label}"
                        substitutions_2[key] = formatted
                    else:
                        key = f"{metric}_{category}_{method}_{time_label}"
                        substitutions_2[key] = "N/A"

    latex_output_2 = get_latex_template_nrmse_fsim().format(**substitutions_2)
    output_file_2 = output_path_obj / "cv_metrics_nrmse_fsim_combined.tex"
    with open(output_file_2, 'w') as f:
        f.write(latex_output_2)
    print(f"Table 2 (NRMSE/FSIM combined) written to: {output_file_2}")

    return resolved_config


def generate_median_metrics_plots(resolved_config: Dict[str, Dict[str, List[Path]]],
                                  output_dir: str,
                                  plot_type: str = 'box',
                                  figsize: Tuple[float, float] = (16, 5),
                                  dpi: int = 300) -> None:
    """
    Generate plots for median metrics.

    Args:
        resolved_config: Configuration with resolved Path objects
        output_dir: Directory to save plots
        plot_type: 'box', 'violin', or 'boxen'
        figsize: Figure size (width, height)
        dpi: DPI for saved figures
    """
    print("\nGenerating median metrics plots...\n")

    # Collect data for median metrics
    df = collect_median_metrics_data(resolved_config)

    if df.empty:
        warnings.warn("No data collected for median metrics plots")
        return

    print(f"Collected {len(df)} data points for plotting")

    # Create output directory
    output_path_obj = Path(output_dir).resolve()
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Generate plot
    plot_filename = f"median_metrics_{plot_type}plot.pdf"
    output_plot_path = output_path_obj / plot_filename

    # Unpack figsize tuple into width and height
    figure_width, figure_height = figsize

    create_median_metrics_plot(
        df=df,
        output_path=output_plot_path,
        plot_type=plot_type,
        figure_width=figure_width,
        figure_height=figure_height,
        figure_dpi=dpi
    )


def main():
    parser = argparse.ArgumentParser(
        description='Process JSON metrics files and generate LaTeX tables and plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s                                    # Generate all tables and the default plot
  %(prog)s -d HCP                             # Specify dataset
  %(prog)s --plot_type violin                 # Use violin plots
  %(prog)s --plot_type boxen                  # Use violin+box overlay plots
  %(prog)s --no_plot                          # Generate only tables
  %(prog)s --figsize 18 6 --dpi 600           # Custom figure settings

Default input:    output/artifacts/LaTeX/cv_metrics.json
Default output:   output/artifacts/metrics/
Default resolved: output/artifacts/metrics/cv_metrics_resolved.json
Default dataset:  HCP

Generates 2 LaTeX tables:
  1. cv_metrics_psnr_ssim_combined.tex - PSNR/SSIM metrics
  2. cv_metrics_nrmse_fsim_combined.tex - NRMSE/FSIM metrics

Generates plots for three median metrics (by default):
  - ODF L2 Norm Error
  - GFA Absolute Error
  - Eigenvector Angular Error

Plot types:
  - box: Standard box plot with quartiles
  - violin: Violin plot showing full distribution
  - boxen: Violin plot with thin box plot overlay (default)

Note: Paths in the JSON config can include wildcards (*, ?, []).
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default='output/artifacts/LaTeX/cv_metrics.json',
        help='Path to JSON configuration file (default: output/artifacts/LaTeX/cv_metrics.json)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output/artifacts/metrics/',
        help='Directory to output LaTeX tables and plots (default: output/artifacts/metrics/)'
    )

    parser.add_argument(
        '-r', '--resolved',
        type=str,
        default='output/artifacts/metrics/cv_metrics_resolved.json',
        help='Path to save resolved paths JSON (default: output/artifacts/metrics/cv_metrics_resolved.json)'
    )

    parser.add_argument(
        '-d', '--dataset',
        type=str,
        default='HCP',
        help='Dataset name to use from configuration (default: HCP)'
    )

    parser.add_argument(
        '-p', '--precision',
        type=int,
        default=3,
        help='Number of decimal places for mean values (default: 3)'
    )

    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip generating plots (only generate LaTeX tables)'
    )

    parser.add_argument(
        '--plot_type',
        type=str,
        choices=['box', 'violin', 'boxen'],
        default='boxen',
        help='Type of plot: box, violin, or boxen (violin+box overlay; default)'
    )

    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[16, 5],
        metavar=('WIDTH', 'HEIGHT'),
        help='Figure size in inches (width height) (default: 16 5)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )

    args = parser.parse_args()

    try:
        # Validate input file exists
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise ConfigurationError(f"Input file '{input_path}' not found.")

        # Load configuration
        print(f"Reading configuration from: {input_path}")
        config = load_paths_config(str(input_path))

        # Generate LaTeX tables and get resolved config
        print("Generating LaTeX tables...\n")
        resolved_config = generate_latex_tables(
            config,
            args.dataset,
            args.output,
            args.resolved,
            args.precision
        )

        # Generate plots unless disabled
        if not args.no_plot:
            generate_median_metrics_plots(
                resolved_config=resolved_config,
                output_dir=args.output,
                plot_type=args.plot_type,
                figsize=tuple(args.figsize),
                dpi=args.dpi
            )

        print("\n✓ Done!")

    except (ConfigurationError, DataProcessingError) as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == '__main__':
    main()

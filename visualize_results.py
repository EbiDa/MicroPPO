#
# Copyright (C) 2024 Daniel Ebi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import itertools
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from typing import Dict, Optional

COLOR_PALETTE = sns.color_palette("tab10")
EXPERIMENTAL_DATA_FOLDER = "output/experiments"
OUTPUT_FOLDER = "output/plots"
N_FOLDS = 10
HORIZON = 168

LINEWITH_IN_PTS = 252
TEXTWITH_IN_PTS = 516

MODEL_PLAINTEXT_LABELS = {
    'milp': 'B1 (Offline optimization)',
    'seq_milp': 'B2 (Sequential optimization)',
    'rb_economic': 'B3 (Rule-based Operation)',
    'rb_own': 'B4 (Self-consumption pattern)',
    'ppo_c': 'B5 (PPO-C)',
    'dqn': 'B6 (DQN)',
    'ppo_d': 'B7 (PPO-D)',
    'microppo': 'MicroPPO',
}

MODEL_PLOT_SETTINGS = {
    'milp': {'color': 0, 'marker': 'o', 'markersize': 4},
    'seq_milp': {'color': 1, 'marker': (4, 1, 45), 'markersize': 6},
    'rb_economic': {'color': 2, 'marker': (4, 1, 0), 'markersize': 6},
    'rb_own': {'color': 9, 'marker': 'h', 'markersize': 4},
    'ppo_c': {'color': 8, 'marker': 's', 'markersize': 4},
    'dqn': {'color': 5, 'marker': '>', 'markersize': 4},
    'ppo_d': {'color': 7, 'marker': 'X', 'markersize': 4},
    'microppo': {'color': 0, 'marker': '^', 'markersize': 4}
}


def load_results(run_id: str, run_date: str, model_id: str):
    def _prepare_data(raw_data: pd.DataFrame, algorithm: Optional[str] = None):
        tmp = raw_data[raw_data['remaining_steps'] == 0]
        tmp.reset_index(inplace=True)
        tmp.loc[:, 'fold_1'] = tmp['fold']
        return tmp

    raw_exp_data = pd.concat([pd.read_csv(
        EXPERIMENTAL_DATA_FOLDER + "/" + run_date + "/" + run_id + "/" + model_id + '/' + model_id + "_" + run_date.replace(
            '-',
            '') + "_" + str(
            i) + ".csv") for i in range(1, N_FOLDS)], ignore_index=True)
    exp_data = _prepare_data(raw_data=raw_exp_data, algorithm=MODEL_PLAINTEXT_LABELS[model_id])
    weeks = np.arange(1, 13)
    time_mapping = {exp_data.loc[i, 'timestamp']: weeks[i] for i in range(len(weeks))}

    res = pd.DataFrame()
    res['Cumulative Reward (€)'] = [exp_data.loc[i, 'cumulative_reward'] for i in range(len(exp_data))]
    res['Algorithm'] = [model_id for i in range(len(res))]
    res['Week'] = [time_mapping[exp_data.loc[i, 'timestamp']] for i in range(len(res))]
    res['Fold'] = [exp_data.loc[i, 'fold'] for i in range(len(res))]
    return res


def get_figure_size(width, fraction=1):
    """Get figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts.
    fraction: float, optional
            fThe desired height of the figure as a fraction of its width.

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Height of figure (in pts)
    height = width * fraction

    # Convert from pts to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = width * inches_per_pt
    # Figure height in inches
    fig_height_in = height * inches_per_pt

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


def plot_mean_episodic_profits(exp_results: Dict[str, pd.DataFrame], run_id: str, run_date: str):
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = 'Times New Roman'
    mpl.rcParams["font.size"] = "8"

    fig, ax1 = plt.subplots(figsize=get_figure_size(width=252, fraction=1.2), dpi=600)

    for algorithm in list(exp_results.keys()):
        sns.lineplot(data=exp_results[algorithm], x='Week', y='Cumulative Reward (€)',
                     marker=MODEL_PLOT_SETTINGS[algorithm]['marker'],
                     errorbar=None, ax=ax1, linestyle='', linewidth=1,
                     markersize=MODEL_PLOT_SETTINGS[algorithm]['markersize'],
                     label=MODEL_PLAINTEXT_LABELS[algorithm],
                     color=COLOR_PALETTE[MODEL_PLOT_SETTINGS[algorithm]['color']], markeredgecolor=None)

    ax1.y_lim = (-20, 5.5)
    ax1.set_yticks(np.arange(-20, 7.5, 2.5))
    y_labels_ax1 = ['' if val % 5 != 0 else f'{val:.1f}' for val in np.arange(-20, 7.5, 2.5)]
    ax1.set_yticklabels(y_labels_ax1, fontsize="7")

    ax1.set_xlabel('Test Week')
    ax1.set_ylabel('Mean Episode Profit (€)')

    ax1.set_xticks(np.arange(1, 13))
    ax1.set_xticklabels(ax1.get_xticks(), fontsize=7)

    sns.move_legend(ax1, "lower center",
                    bbox_to_anchor=(0.45, 1.02), ncol=2, title='Approach', frameon=False, fontsize='7',
                    title_fontsize='8')

    ax2 = ax1.inset_axes([0.3, 0.1, .35, .35])
    for algorithm in list(exp_results.keys()):
        sns.lineplot(data=exp_results[algorithm], x='Week', y='Cumulative Reward (€)',
                     marker=MODEL_PLOT_SETTINGS[algorithm]['marker'],
                     errorbar=None, ax=ax2, linestyle='', linewidth=1,
                     markersize=MODEL_PLOT_SETTINGS[algorithm]['markersize'],
                     label=MODEL_PLAINTEXT_LABELS[algorithm],
                     color=COLOR_PALETTE[MODEL_PLOT_SETTINGS[algorithm]['color']], markeredgecolor=None)

    ax2.set(xlim=(4.5, 8.5), ylim=(0.4, 4.6))
    ax2.get_legend().remove()
    ax2.set_yticks(np.arange(0.5, 4.75, 0.25))
    y_labels = ['' if val % 1 != 0 else f'{val:.1f}' for val in np.arange(0.5, 4.75, 0.25)]
    ax2.set_yticklabels(y_labels, fontsize="7")
    ax2.set_xticks([5, 6, 7, 8])
    ax2.set_xticklabels(ax2.get_xticks(), fontsize=7)

    ax2.set(xlabel=None, ylabel=None)
    ax1.indicate_inset_zoom(ax2, edgecolor="black")

    plt.grid()
    plt.tight_layout()

    output_path = OUTPUT_FOLDER + '/' + run_date + '/' + run_id
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except FileExistsError:
        print("Output folder already exists.")

    plt.savefig(output_path + '/' + run_id + '_evaluation_mean_episode_profits.pdf', dpi=600, bbox_inches='tight')
    print("Plot was successfully saved to the disk.")


def print_mean_episode_profit(exp_results: Dict[str, pd.DataFrame], week: int):
    for algorithm in list(exp_results.keys()):
        print(MODEL_PLAINTEXT_LABELS[algorithm] + ': ' + str(
            np.mean(exp_results[algorithm][exp_results[algorithm]['Week'] == week]['Cumulative Reward (€)'])))


def plot_optimality_gaps(exp_results: Dict[str, pd.DataFrame], reference_model_id: str, run_id: str, run_date: str):
    algorithms = list(exp_results.keys())
    algorithms.remove(reference_model_id)
    for algorithm in algorithms:
        exp_results[algorithm]['Relative Improvement'] = - ((exp_results[algorithm]['Cumulative Reward (€)'] -
                                                             exp_results[reference_model_id][
                                                                 'Cumulative Reward (€)']) / abs(
            exp_results[reference_model_id]['Cumulative Reward (€)']))

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = 'Times New Roman'
    mpl.rcParams["font.size"] = "8"

    plot_data = pd.DataFrame()
    for algorithm in algorithms:
        plot_data = pd.concat([plot_data, exp_results[algorithm]], axis=0, ignore_index=True)

    plot_data['Relative Improvement'] = plot_data['Relative Improvement'].clip(0, 100)

    fig, ax1 = plt.subplots(figsize=get_figure_size(width=252, fraction=1.0), dpi=600)
    boxplot = sns.boxplot(data=plot_data, x='Algorithm', y='Relative Improvement',
                          palette=[COLOR_PALETTE[MODEL_PLOT_SETTINGS[algorithm]['color']] for algorithm in
                                   algorithms], ax=ax1, linewidth=0.8, showfliers=False)

    medians = plot_data.groupby(['Algorithm'], sort=False)['Relative Improvement'].median()

    x_labels_box = []
    for x_tick in range(0, len(algorithms)):
        x_labels_box.append(MODEL_PLAINTEXT_LABELS[plot_data['Algorithm'].unique().tolist()[x_tick]].split(" ")[0])

    plt.xticks(rotation=45)
    ax1.set(ylabel='Optimality Gap', xlabel='Approach')
    plt.ylim(-0.25, 4.5)
    ax1.set_yticks(np.arange(0, 4.5, 0.5))
    y_labels_box = ['' if val % 1 != 0 else f'{val:.1f}' for val in np.arange(0, 4.5, 0.5)]
    ax1.set_yticklabels(y_labels_box, fontsize="7")

    lines = ax1.get_lines()
    categories = ax1.get_xticks()

    for cat in categories:
        offset = 0.08
        if cat == 0:
            offset += 0.03
        elif cat == 3:
            offset += 0.33
        elif cat == 6:
            offset += 0.11
        med = np.round(medians[plot_data['Algorithm'].unique().tolist()[cat]], 4)
        ax1.text(
            cat,
            med + offset,
            f'{med}',
            ha='center',
            va='center',
            size=6,
            fontweight='semibold'
        )

    ax1.set_xticklabels(x_labels_box, fontsize="7", ha='right', rotation_mode='anchor')
    plt.grid()
    fig.tight_layout()

    output_path = OUTPUT_FOLDER + '/' + run_date + '/' + run_id
    try:
        if not os.path.exists(output_path):
            os.makesdirs(output_path)
    except FileExistsError:
        print("Output folder already exists.")

    plt.savefig(output_path + '/' + run_id + '_optimality_gaps.pdf', dpi=600, bbox_inches='tight')
    print("Plot was successfully saved to the disk.")

    if __name__ == "__main__":
        run_id = "example-run"
        run_date = "example"
        models = ['milp', 'seq_milp', 'rb_economic', 'rb_own', 'ppo_c', 'dqn', 'ppo_d', 'microppo']
        results = {model_id: load_results(run_id=run_id, run_date=run_date, model_id=model_id) for model_id in models}

        print("+++ PLOT MEAN EPISODE PROFITS +++")
        plot_mean_episodic_profits(exp_results=results, run_id=run_id, run_date=run_date)

        week = 6
        print("\n+++ PRINT MEAN EPISODE PROFITS FOR TEST WEEK " + str(week) + " +++")
        print_mean_episode_profit(exp_results=results, week=week)

        print("\n+++ PLOT OPTIMALITY GAPS +++")
        plot_optimality_gaps(exp_results=results, reference_model_id='milp', run_id=run_id, run_date=run_date)

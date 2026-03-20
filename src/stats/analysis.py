"""
Statistical analysis and plotting functions for the imprecision experiment.

Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
Linguistics Vanguard, 8(1), 113–127. https://doi.org/10.1515/lingvan-2022-0035
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_LABELS = {
    0: 's₃₀',
    1: 's₃₀±₁',
    2: 's₃₀±₂',
    3: 's₃₀±₃',
    4: 's₃₀±₄',
    5: 's₃₀±₅',
    8: 'sᵢₙ'
}

STATE_ORDER = [0, 1, 2, 3, 4, 5, 8]

# Columns collapsed to match Figure 2 in the paper.
# Interval variants (answerC 8, 9, 18, 19) are merged into a single vᵢₙ column.
COLLAPSED_COLS = {
    'va₃₀':   [10],           # approximator + 8:30
    'v₃₀':    [0],            # bare 8:30
    'v₃₀±₁':  [1],            # bare 8:29 / 8:31
    'v₃₀±₂':  [2],            # bare 8:28 / 8:32
    'v₃₀±₃':  [3],            # bare 8:27 / 8:33
    'v₃₀±₄':  [4],            # bare 8:26 / 8:34
    'v₃₀±₅':  [5],            # bare 8:25 / 8:35
    'va₃₀±₅': [15],           # approximator + 8:25/8:35
    'vᵢₙ':    [8, 9, 18, 19]  # all interval expressions collapsed
}

MOTIVE_COLS = [
    'Precision', 'Accuracy', 'Info lack', 'Misinfo', 'Safe',
    'H needs', 'Context', 'S ease', 'H ease', 'Habit', 'Sound', 'Other'
]

MOTIVE_LABELS = [
    'Level of precision/detail', 'Accuracy/truthfulness',
    'Possible lack of information', 'Possible misinformation',
    'Safe choice', 'Hearer needs', 'Appropriateness for context',
    'Speaker ease', 'Hearer ease', 'Habit/convention',
    'How it sounds', 'Other/irrelevant'
]


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def build_matrix(df, context, normalize=True):
    """Build a response matrix for a given context.

    Rows are information states; columns are collapsed response categories
    matching Figure 2 in Mühlenbernd & Solt (2022).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned participant-level data (answerC != -1).
    context : str
        'police' or 'neighbor'.
    normalize : bool
        If True, return row-normalized proportions (0–1).
        If False, return absolute counts.

    Returns
    -------
    pd.DataFrame
        Shape (7, 9): rows = information states, columns = response categories.
    """
    sub = df[df['context'] == context]
    rows = {}
    for s in STATE_ORDER:
        state_sub = sub[sub['stateC'] == s]
        n = len(state_sub)
        row = {
            col: state_sub['answerC'].isin(codes).sum() / (n if normalize and n > 0 else 1)
            for col, codes in COLLAPSED_COLS.items()
        }
        rows[STATE_LABELS[s]] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_matrix(matrix, title, ax):
    """Plot a response matrix as a colour-coded heatmap with % annotations.

    Parameters
    ----------
    matrix : pd.DataFrame
        Normalized response matrix (values 0–1).
    title : str
        Axes title.
    ax : matplotlib.axes.Axes
        Target axes.

    Returns
    -------
    matplotlib.image.AxesImage
        The image object (for attaching a shared colorbar).
    """
    cmap = plt.cm.YlGnBu
    im = ax.imshow(matrix.values, aspect='equal', cmap=cmap, vmin=0, vmax=1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            pct = int(round(val * 100))
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, str(pct), ha='center', va='center',
                    fontsize=8, color=color)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('response expression', fontsize=10)
    ax.set_ylabel('information state', fontsize=10)
    return im


def plot_response_matrices(matrix_police, matrix_neighbor,
                           title=None, save_path=None):
    """Plot police and neighbor response matrices side by side.

    Parameters
    ----------
    matrix_police : pd.DataFrame
        Normalized response matrix for the police context.
    matrix_neighbor : pd.DataFrame
        Normalized response matrix for the neighbor context.
    title : str, optional
        Figure suptitle.
    save_path : str, optional
        If provided, save figure to this path.
    """
    cell_size = 0.55
    n_rows, n_cols = 7, 9
    ax_w = n_cols * cell_size
    ax_h = n_rows * cell_size
    fig_w = ax_w * 2 + 2.5
    fig_h = ax_h + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    left_margin = 0.08
    right_margin = 0.88
    gap = 0.06
    ax_width = (right_margin - left_margin - gap) / 2
    bottom, top = 0.18, 0.88
    ax_height = top - bottom

    ax1 = fig.add_axes([left_margin, bottom, ax_width, ax_height])
    ax2 = fig.add_axes([left_margin + ax_width + gap, bottom, ax_width, ax_height])
    cax = fig.add_axes([0.90, bottom, 0.02, ax_height])

    plot_matrix(matrix_police, '(a) police context', ax1)
    im2 = plot_matrix(matrix_neighbor, '(b) neighbor context', ax2)

    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    if title:
        fig.suptitle(title, y=0.97, fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    plt.show()


def plot_motives(df, save_path=None):
    """Plot participant motive categories by context (Table A.3).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned participant-level data.
    save_path : str, optional
        If provided, save figure to this path.
    """
    motive_counts = {
        ctx: [df[df['context'] == ctx][col].notna().sum() for col in MOTIVE_COLS]
        for ctx in ['police', 'neighbor']
    }
    motive_df = pd.DataFrame(motive_counts, index=MOTIVE_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MOTIVE_LABELS))
    width = 0.38

    ax.bar(x - width / 2, motive_df['police'], width,
           label='Police context', color='steelblue', alpha=0.85)
    ax.bar(x + width / 2, motive_df['neighbor'], width,
           label='Neighbor context', color='coral', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MOTIVE_LABELS, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Number of respondents')
    ax.set_title('Participant motive categories by context (Table A.3)',
                 fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()

    return motive_df


def plot_rounding_effect(ct, save_path=None):
    """Plot round vs. non-round response proportions by context.

    Parameters
    ----------
    ct : pd.DataFrame
        Contingency table with rows = contexts and columns ['non-round', 'round'],
        as returned by pd.crosstab(sub['context'], sub['rounded']).
    save_path : str, optional
        If provided, save figure to this path.
    """
    ct_prop = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))

    x = np.arange(len(ct_prop.index))
    width = 0.35
    colors = {'round': '#4a6fa5', 'non-round': '#b0c4de'}

    bars_round    = ax.bar(x - width / 2, ct_prop['round'],     width,
                           label='round',     color=colors['round'])
    bars_nonround = ax.bar(x + width / 2, ct_prop['non-round'], width,
                           label='non-round', color=colors['non-round'])

    for bar in [*bars_round, *bars_nonround]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f'{h:.0%}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(ct_prop.index)
    ax.set_ylabel('Proportion of responses')
    ax.set_xlabel('Context')
    ax.set_title('Round vs. non-round responses by context\n'
                 '(non-round precise states s₃₀±₁ to s₃₀±₄)')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def chi2_context_test(df, states):
    """Chi-square test of context independence for a subset of states.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned participant-level data.
    states : list of int
        stateC values to include.

    Returns
    -------
    tuple : (chi2, p, dof)
    """
    sub = df[df['stateC'].isin(states)]
    ct = pd.crosstab(sub['context'], sub['answerC'])
    ct = ct.loc[:, (ct != 0).any(axis=0)]
    chi2, p, dof, _ = chi2_contingency(ct)
    return chi2, p, dof


def chi2_per_state(df):
    """Run chi-square context tests for each information state individually.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned participant-level data.

    Returns
    -------
    pd.DataFrame
        Columns: state, chi2, dof, p, sig
    """
    results = []
    for state_code in STATE_ORDER:
        sub = df[df['stateC'] == state_code]
        ct = pd.crosstab(sub['context'], sub['answerC'])
        ct = ct.loc[:, (ct != 0).any(axis=0)]
        if ct.shape[1] < 2:
            results.append({
                'state': STATE_LABELS[state_code],
                'chi2': None, 'dof': None, 'p': None, 'sig': '—'
            })
            continue
        try:
            chi2, p, dof, _ = chi2_contingency(ct)
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            results.append({
                'state': STATE_LABELS[state_code],
                'chi2': round(chi2, 3),
                'dof': dof,
                'p': round(p, 4),
                'sig': sig
            })
        except ValueError:
            results.append({
                'state': STATE_LABELS[state_code],
                'chi2': None, 'dof': None, 'p': None, 'sig': 'error'
            })
    return pd.DataFrame(results).set_index('state')

"""
Experiment configuration and empirical data for the imprecision study.

Defines the game parameters for the imprecision experiment reported in:

    Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
    Linguistics Vanguard, 8(1), 113–127.
    https://doi.org/10.1515/lingvan-2022-0035

This module provides:
- build_imprecision_game(): constructs the Game object for the experiment
- Empirical response matrices from the production study (Figure 2)

Empirical matrices are used for model evaluation (fitting w_R and w_A)
and for comparison with model predictions (Figure 4).
"""

from .game import Game
from . import utils


# ---------------------------------------------------------------------------
# Model parameters (Table in Technical Report)
# ---------------------------------------------------------------------------

# Default model parameters from the paper
LAM   = 5.0   # softmax parameter λ
ALPHA = 1.0   # imprecision parameter α
COST_COMPLEX = 0.1  # utterance cost for approximator/interval forms

# Optimal weight parameters per context (Section 4)
W_R_POLICE   = 0.46   # roundness weight, police context
W_A_POLICE   = 0.22   # accuracy weight, police context
W_R_NEIGHBOR = 0.66   # roundness weight, neighbor context
W_A_NEIGHBOR = 0.02   # accuracy weight, neighbor context


# ---------------------------------------------------------------------------
# Game setup
# ---------------------------------------------------------------------------

def build_imprecision_game(alpha=ALPHA):
    """Construct the signaling game for the imprecision experiment.

    Information states S represent the 12 clock times tested in the
    experiment (7 distinct categories, with ±1 through ±5 collapsed to
    singleton representative values for the model).

    Utterances V cover bare time expressions, approximator-modified
    expressions, and interval expressions.

    Parameters
    ----------
    alpha : float
        Imprecision parameter α for the payoff function π.
        Defaults to 1.0 (value used in the paper).

    Returns
    -------
    Game
        Configured game object for the imprecision experiment.
    """

    # Information states S: each state is a list of time points (minutes)
    # Singleton states represent precise clock readings;
    # the last state represents the approximate range 8:26–8:34
    S = [
        [25], [26], [27], [28], [29], [30],
        [31], [32], [33], [34], [35],
        [26, 27, 28, 29, 30, 31, 32, 33, 34]   # approximate state sᵢₙ
    ]

    # Utterances V
    # v25–v35: bare precise expressions ("It happened at 8:XX")
    # vIn:     interval expression ("between 8:26 and 8:34")
    # vA30:    approximator + round value ("around 8:30")
    # vA25:    approximator + lower bound ("around 8:25")
    # vA35:    approximator + upper bound ("around 8:35")
    V = [
        'v25', 'v26', 'v27', 'v28', 'v29', 'v30',
        'v31', 'v32', 'v33', 'v34', 'v35',
        'vIn', 'vA30', 'vA25', 'vA35'
    ]

    # Denotation function D: [[v]] ⊆ T
    # Start with 1-to-1 literal denotation for bare expressions,
    # then override for approximator and interval utterances
    D = utils.literal_denotation(S, V)
    D['vIn']  = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    D['vA30'] = [27, 28, 29, 30, 31, 32, 33]
    D['vA25'] = [22, 23, 24, 25, 26, 27, 28]
    D['vA35'] = [32, 33, 34, 35, 36, 37, 38]

    # Roundness hierarchy (Table A.4 in the paper)
    # Level 0 (least round): non-round non-5 values + interval
    # Level 1 (intermediate): multiples of 5 (excl. 30) + approximators
    # Level 2 (most round):   8:30 and approximator + 8:30
    roundness_hierarchy = [
        ['v26', 'v27', 'v28', 'v29', 'v31', 'v32', 'v33', 'v34', 'vIn'],
        ['v25', 'vA25', 'vA35', 'v35'],
        ['v30', 'vA30']
    ]

    # Uniform prior over S
    Pr = [1.0 / len(S)] * len(S)

    # Utterance costs: 0 for bare expressions, COST_COMPLEX for longer forms
    C = [0.0] * len(V)
    for i, v in enumerate(V):
        if v in ('vIn', 'vA30', 'vA25', 'vA35'):
            C[i] = COST_COMPLEX

    game = Game(S=S, V=V, Pr=Pr, D=D, C=C, alpha=alpha)
    game.roundness_hierarchy = roundness_hierarchy

    return game


# ---------------------------------------------------------------------------
# Empirical response matrices (Figure 2, Mühlenbernd & Solt 2022)
# ---------------------------------------------------------------------------
# Rows: information states [s30, s30±1, s30±2, s30±3, s30±4, s30±5, sIn]
# Cols: response categories [v30, v30±1, v30±2, v30±3, v30±4, v30±5, vIn, vA30, vA35]

#: Normalized empirical matrix for the police context (Figure 2a)
EMPIRICAL_POLICE = [
    [0.97, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.00],  # s30
    [0.19, 0.78, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.00],  # s30±1
    [0.07, 0.00, 0.81, 0.00, 0.00, 0.00, 0.00, 0.11, 0.00],  # s30±2
    [0.15, 0.00, 0.00, 0.73, 0.00, 0.03, 0.00, 0.06, 0.03],  # s30±3
    [0.00, 0.00, 0.00, 0.00, 0.75, 0.18, 0.00, 0.00, 0.07],  # s30±4
    [0.00, 0.00, 0.00, 0.03, 0.00, 0.90, 0.00, 0.03, 0.03],  # s30±5
    [0.30, 0.00, 0.00, 0.00, 0.00, 0.02, 0.29, 0.39, 0.00],  # sIn
]

#: Normalized empirical matrix for the neighbor context (Figure 2b)
EMPIRICAL_NEIGHBOR = [
    [0.94, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.00],  # s30
    [0.41, 0.41, 0.03, 0.00, 0.00, 0.00, 0.00, 0.16, 0.00],  # s30±1
    [0.26, 0.00, 0.53, 0.00, 0.00, 0.00, 0.00, 0.18, 0.03],  # s30±2
    [0.13, 0.00, 0.00, 0.61, 0.00, 0.06, 0.00, 0.16, 0.03],  # s30±3
    [0.06, 0.00, 0.00, 0.00, 0.59, 0.22, 0.00, 0.06, 0.06],  # s30±4
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],  # s30±5
    [0.47, 0.00, 0.00, 0.00, 0.02, 0.00, 0.13, 0.36, 0.02],  # sIn
]

#: Row labels for the 7 information state categories
STATE_LABELS = ['s₃₀', 's₃₀±₁', 's₃₀±₂', 's₃₀±₃', 's₃₀±₄', 's₃₀±₅', 'sᵢₙ']

#: Column labels for the 9 response categories
RESPONSE_LABELS = ['v₃₀', 'v₃₀±₁', 'v₃₀±₂', 'v₃₀±₃', 'v₃₀±₄', 'v₃₀±₅', 'vᵢₙ', 'va₃₀', 'va₃₀±₅']

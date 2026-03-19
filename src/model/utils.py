"""
Utility functions for the Imprecision Model.

Implements mathematical operations referenced in:

    Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
    Linguistics Vanguard, 8(1), 113–127.
    https://doi.org/10.1515/lingvan-2022-0035
"""

import math


# ---------------------------------------------------------------------------
# Distance and similarity
# ---------------------------------------------------------------------------

def dis(s1, s2):
    """Compute the average item-by-item minimal distance between two
    information states s1 and s2.

    This implements the distance function dis(s, s') from the paper
    (Section 3.2.1), used to define the payoff function π(s, s').

    Each information state is a list of time points. The distance is the
    average of the minimal distances from each point in s1 to s2 and
    vice versa.

    Parameters
    ----------
    s1, s2 : list of int
        Information states represented as lists of time points.

    Returns
    -------
    float
        Average minimal distance between s1 and s2.
    """
    avg_distance = 0.0
    n = len(s1) + len(s2)

    for t1 in s1:
        min_diff = min(abs(t1 - t2) for t2 in s2)
        avg_distance += min_diff / n

    for t2 in s2:
        min_diff = min(abs(t2 - t1) for t1 in s1)
        avg_distance += min_diff / n

    return avg_distance


def similarity(s1, s2, alpha):
    """Compute the Nosofsky perceptual similarity between two information
    states, controlled by imprecision parameter alpha.

    Implements π(s, s') = exp(-dis(s, s')² / α²) from the paper
    (Section 3.2.1). See also Franke & Correia (2018), p. 1050.

    Parameters
    ----------
    s1, s2 : list of int
        Information states.
    alpha : float
        Imprecision parameter α > 0. Higher values reduce the impact
        of distance on similarity (more tolerance for imprecision).

    Returns
    -------
    float
        Similarity value in [0, 1].
    """
    return math.exp(-(dis(s1, s2) ** 2) / (alpha ** 2))


# ---------------------------------------------------------------------------
# Probabilistic inference
# ---------------------------------------------------------------------------

def literal_belief(game, s, v):
    """Return the literal hearer's belief probability for state s given
    utterance v, based solely on the denotation function.

    Parameters
    ----------
    game : Game
        The signaling game.
    s : list of int
        An information state in game.S.
    v : str
        An utterance in game.V.

    Returns
    -------
    float
        1 / |[[v]]| if s ∈ [[v]], else 0.
    """
    if s in game.D[v]:
        return 1.0 / len(game.D[v])
    return 0.0


def bayes_update(game, s, v, speaker_strategy):
    """Compute the Bayesian update of hearer belief for state s given
    utterance v and a speaker strategy.

    Implements P_H(s|v) ∝ Pr(s) · P_S0(v|s) from the paper
    (Section 3.2.1). Falls back to literal belief if the denominator
    is zero.

    Parameters
    ----------
    game : Game
        The signaling game.
    s : list of int
        An information state in game.S.
    v : str
        An utterance in game.V.
    speaker_strategy : SpeakerStrategy
        The speaker strategy used to compute the update.

    Returns
    -------
    float
        Posterior probability P_H(s|v).
    """
    denominator = sum(
        speaker_strategy.get_entry(s2, v) * game.Pr[game.S.index(s2)]
        for s2 in game.S
    )

    if denominator > 0.0:
        prior = game.Pr[game.S.index(s)]
        return speaker_strategy.get_entry(s, v) * prior / denominator
    else:
        return literal_belief(game, s, v)


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def normalize_rows(matrix):
    """Normalize a 2D matrix so that each row sums to 1.

    Parameters
    ----------
    matrix : list of list of float

    Returns
    -------
    list of list of float
        Row-normalized matrix.
    """
    return [
        [val / row_sum for val in row]
        for row in matrix
        if (row_sum := sum(row)) > 0
    ]


def select_submatrix(matrix, row_indices, col_indices):
    """Extract a submatrix by selecting specific rows and columns.

    Used to reduce the full model output matrix to the 7×9 display
    matrix matching Figure 2 in the paper.

    Parameters
    ----------
    matrix : list of list of float
    row_indices : list of int
    col_indices : list of int

    Returns
    -------
    list of list of float
    """
    return [
        [matrix[r][c] for c in col_indices]
        for r in row_indices
    ]


# ---------------------------------------------------------------------------
# Game setup helpers
# ---------------------------------------------------------------------------

def literal_denotation(S, V):
    """Build a 1-to-1 denotation function mapping each utterance to its
    corresponding singleton information state.

    Produces the initial truthful speaker denotation used in P_S0,
    where each utterance v_i is true only in state s_i.

    Parameters
    ----------
    S : list of list of int
        Information states.
    V : list of str
        Utterances.

    Returns
    -------
    dict
        Mapping from utterance (str) to information state (list of int).
    """
    return {V[i]: S[i] for i in range(min(len(S), len(V)))}


def is_subset(list1, list2):
    """Check whether all elements of list1 are contained in list2.

    Parameters
    ----------
    list1, list2 : list

    Returns
    -------
    bool
    """
    return all(entry in list2 for entry in list1)

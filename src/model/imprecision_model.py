"""
The Imprecision Model: a probabilistic game-theoretic speaker model.

Implements the model from:

    Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
    Linguistics Vanguard, 8(1), 113–127.
    https://doi.org/10.1515/lingvan-2022-0035

The model predicts the probability P_S(v | s) that a pragmatic speaker
uses utterance v given information state s. The speaker maximises a
total utility U_tot via a softmax choice rule:

    P_S(v | s) ∝ exp(λ · U_tot(v, s, P_H, w))

where U_tot decomposes into three components:

    U_tot(v, s, P_H, w) = U_inf(v, s, P_H)
                        + w_R · U_rnd(v)
                        + w_A · U_acc(v, s)
                        - C(v)

See the paper (Section 3) for full formal definitions.
"""

import math
from .strategies import SpeakerStrategy, HearerStrategy
from . import utils


class ImprecisionModel:
    """The Imprecision Model as defined in Mühlenbernd & Solt (2022).

    Parameters
    ----------
    game : Game
        The signaling game G = (S, V, Pr, D, C, π).
    w_R : float
        Weight for the roundness utility U_rnd. Represents the speaker's
        pressure towards hearer-oriented simplification.
    w_A : float
        Weight for the accuracy utility U_acc. Represents the speaker's
        pressure towards accuracy.
    lam : float
        Softmax parameter λ. Controls how sharply the speaker maximises
        utility (higher = more deterministic).
    roundness_hierarchy : list of list of str
        Nested list of utterances ordered by roundness level, from least
        to most round. E.g. [['v26', ...], ['v25', 'v35'], ['v30', 'vA30']].
        Defaults to a flat hierarchy (all utterances equally round).
    """

    def __init__(self, game, w_R=0.0, w_A=0.0, lam=1000.0,
                 roundness_hierarchy=None):
        self.game = game
        self.w_R = w_R
        self.w_A = w_A
        self.lam = lam

        if roundness_hierarchy is not None:
            self.roundness_hierarchy = roundness_hierarchy
        else:
            # Flat hierarchy: all utterances at level 0
            self.roundness_hierarchy = [list(game.V)]

    # -----------------------------------------------------------------------
    # Main inference pipeline
    # -----------------------------------------------------------------------

    def P_S0(self):
        """Compute the truthful (literal) speaker strategy P_S0.

        P_S0 assigns uniform probability over all utterances v whose
        core-semantic meaning [[v]] contains (is a superset of) the
        current information state s.

        Returns
        -------
        SpeakerStrategy
            The literal speaker strategy.
        """
        strategy = SpeakerStrategy(self.game)

        for s in self.game.S:
            valid = [v for v in self.game.V
                     if utils.is_subset(s, self.game.D[v])]
            prob = 1.0 / len(valid) if valid else 0.0

            for v in self.game.V:
                value = prob if utils.is_subset(s, self.game.D[v]) else 0.0
                strategy.set_entry(s, v, value)

        return strategy

    def P_H(self, speaker_strategy):
        """Compute the hearer's Bayesian interpretation strategy P_H,
        given a speaker strategy.

        P_H(s|v) ∝ Pr(s) · P_S0(v|s)

        Parameters
        ----------
        speaker_strategy : SpeakerStrategy

        Returns
        -------
        HearerStrategy
        """
        strategy = HearerStrategy(self.game)

        for v in self.game.V:
            for s in self.game.S:
                value = utils.bayes_update(self.game, s, v, speaker_strategy)
                strategy.set_entry(v, s, value)

        return strategy

    def P_S(self, hearer_strategy):
        """Compute the pragmatic speaker strategy P_S via softmax over
        total utility, given hearer strategy P_H.

        P_S(v | s) ∝ exp(λ · U_tot(v, s, P_H, w))

        Parameters
        ----------
        hearer_strategy : HearerStrategy

        Returns
        -------
        SpeakerStrategy
            The pragmatic speaker strategy.
        """
        matrix = [
            [math.exp(self.lam * self.U_tot(v, s, hearer_strategy))
             for v in self.game.V]
            for s in self.game.S
        ]
        return SpeakerStrategy(self.game, matrix)

    def run(self):
        """Run the full inference pipeline P_S0 → P_H → P_S.

        Convenience method returning the pragmatic speaker strategy.

        Returns
        -------
        SpeakerStrategy
        """
        S0 = self.P_S0()
        H  = self.P_H(S0)
        S1 = self.P_S(H)
        return S1

    # -----------------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------------

    def U_tot(self, v, s, P_H):
        """Total speaker utility U_tot(v, s, P_H, w).

        U_tot = U_inf(v, s, P_H) + w_R · U_rnd(v) + w_A · U_acc(v, s) - C(v)

        Parameters
        ----------
        v : str
        s : list of int
        P_H : HearerStrategy

        Returns
        -------
        float
        """
        return (
            self.U_inf(v, s, P_H)
            + self.w_R * self.U_rnd(v)
            + self.w_A * self.U_acc(v, s)
            - self.game.cost(v)
        )

    def U_inf(self, v, s, P_H):
        """Informational utility U_inf(v, s, P_H).

        U_inf(v, s, P_H) = Σ_{s'∈S} P_H(s'|v) · π(s, s')

        Parameters
        ----------
        v : str
        s : list of int
        P_H : HearerStrategy

        Returns
        -------
        float
        """
        return sum(
            P_H.get_entry(v, s_prime) * self.game.payoff(s, s_prime)
            for s_prime in self.game.S
        )

    def U_rnd(self, v):
        """Roundness utility U_rnd(v).

        U_rnd(v) = level(v) / (n_levels - 1)

        where level(v) is the 0-indexed position of v in
        roundness_hierarchy (higher index = rounder). Returns 0 for a
        flat hierarchy.

        Parameters
        ----------
        v : str

        Returns
        -------
        float
            Value in [0, 1].
        """
        n_levels = len(self.roundness_hierarchy)
        if n_levels <= 1:
            return 0.0

        for level, utterances in enumerate(self.roundness_hierarchy):
            if v in utterances:
                return level / (n_levels - 1.0)

        return 0.0

    def U_acc(self, v, s):
        """Accuracy utility U_acc(v, s).

        U_acc(v, s) = |[[v]] ∩ s| / |s|

        The probability that the literal interpretation of v is true
        given information state s.

        Parameters
        ----------
        v : str
        s : list of int

        Returns
        -------
        float
            Value in [0, 1].
        """
        overlap = sum(1 for t in s if t in self.game.D[v])
        return overlap / len(s)

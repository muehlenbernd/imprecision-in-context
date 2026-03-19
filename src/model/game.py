"""
Signaling game container for the Imprecision Model.

Implements the game structure G = (S, V, Pr, D, C, π) as defined in:

    Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
    Linguistics Vanguard, 8(1), 113–127.
    https://doi.org/10.1515/lingvan-2022-0035
"""

from . import utils


class Game:
    """A signaling game G = (S, V, Pr, D, C, π).

    Attributes
    ----------
    S : list of list of int
        Set of information states. Each state is a list of time points,
        e.g. [30] for the exact value 8:30 or [26,27,...,34] for the
        approximate range.
    V : list of str
        Set of utterances, e.g. ['v30', 'vA30', 'vIn'].
    Pr : list of float
        Prior probability distribution over S. Defaults to uniform.
    D : dict
        Denotation function mapping each utterance v ∈ V to its
        core-semantic meaning [[v]] ⊆ T (a list of time points).
    C : list of float
        Utterance costs C(v) for each v ∈ V. Simple utterances default
        to 0; complex forms (e.g. approximator phrases) carry a small
        positive cost.
    pi : list of list of float
        Payoff matrix π(s, s') computed via the Nosofsky similarity
        function, controlled by imprecision parameter α.
    """

    def __init__(self, S, V, Pr=None, D=None, C=None, alpha=0.0):
        """
        Parameters
        ----------
        S : list of list of int
            Information states.
        V : list of str
            Utterances.
        Pr : list of float, optional
            Prior probabilities. Defaults to uniform over S.
        D : dict, optional
            Denotation function. Defaults to trivial (every utterance
            true in every state).
        C : list of float, optional
            Utterance costs. Defaults to 0 for all utterances.
        alpha : float
            Imprecision parameter α for the payoff function π.
        """
        self.S = S
        self.V = V

        # Prior: uniform by default
        self.Pr = Pr if Pr is not None else [1.0 / len(S)] * len(S)

        # Denotation: trivially true in all states by default
        self.D = D if D is not None else {v: S for v in V}

        # Costs: zero by default
        self.C = C if C is not None else [0.0] * len(V)

        # Payoff matrix π(s, s') via Nosofsky similarity
        self.pi = [
            [utils.similarity(s, s_prime, alpha) for s_prime in S]
            for s in S
        ]

    def cost(self, v):
        """Return the utterance cost C(v) for utterance v.

        Parameters
        ----------
        v : str
            An utterance in V.

        Returns
        -------
        float
        """
        return self.C[self.V.index(v)]

    def payoff(self, s, s_prime):
        """Return the payoff π(s, s') for information states s and s'.

        Parameters
        ----------
        s, s_prime : list of int
            Information states in S.

        Returns
        -------
        float
        """
        return self.pi[self.S.index(s)][self.S.index(s_prime)]

    def __repr__(self):
        return (
            f"Game(|S|={len(self.S)}, |V|={len(self.V)}, "
            f"Pr={'uniform' if len(set(self.Pr)) == 1 else 'custom'})"
        )

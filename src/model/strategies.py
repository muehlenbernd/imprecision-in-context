"""
Speaker and hearer strategy containers for the Imprecision Model.

Both strategies wrap a 2D probability matrix with named accessors.
They are implementation details of the model and are not referenced
directly in the paper's formal exposition.

    Mühlenbernd, R. & Solt, S. (2022). Modeling (im)precision in context.
    Linguistics Vanguard, 8(1), 113–127.
    https://doi.org/10.1515/lingvan-2022-0035
"""


class SpeakerStrategy:
    """A speaker production strategy P_S.

    Wraps a matrix X where X[s_index][v_index] = P(v | s) —
    the probability that the speaker uses utterance v in state s.

    Parameters
    ----------
    game : Game
        The signaling game (provides S and V).
    X : list of list of float, optional
        Production matrix. Initialized to all zeros if not provided.
    """

    def __init__(self, game, X=None):
        self.S = game.S
        self.V = game.V
        self.X = X if X is not None else [
            [0.0] * len(game.V) for _ in game.S
        ]

    def get_entry(self, s, v):
        """Return P(v | s).

        Parameters
        ----------
        s : list of int
            Information state.
        v : str
            Utterance.

        Returns
        -------
        float
        """
        return self.X[self.S.index(s)][self.V.index(v)]

    def set_entry(self, s, v, value):
        """Set P(v | s) = value.

        Parameters
        ----------
        s : list of int
        v : str
        value : float
        """
        self.X[self.S.index(s)][self.V.index(v)] = value


class HearerStrategy:
    """A hearer perception strategy P_H.

    Wraps a matrix X where X[v_index][s_index] = P(s | v) —
    the probability that the hearer interprets utterance v as state s.

    Parameters
    ----------
    game : Game
        The signaling game (provides S and V).
    X : list of list of float, optional
        Perception matrix. Initialized to all zeros if not provided.
    """

    def __init__(self, game, X=None):
        self.S = game.S
        self.V = game.V
        self.X = X if X is not None else [
            [0.0] * len(game.S) for _ in game.V
        ]

    def get_entry(self, v, s):
        """Return P(s | v).

        Parameters
        ----------
        v : str
            Utterance.
        s : list of int
            Information state.

        Returns
        -------
        float
        """
        return self.X[self.V.index(v)][self.S.index(s)]

    def set_entry(self, v, s, value):
        """Set P(s | v) = value.

        Parameters
        ----------
        v : str
        s : list of int
        value : float
        """
        self.X[self.V.index(v)][self.S.index(s)] = value

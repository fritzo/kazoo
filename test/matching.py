#!/usr/bin/python

"""
A general framework for solving matching problems via belief propagation.

TODO implement example subtypes of Matching
TODO add examples and unit tests
TODO develop measures of goodness-of-fit
"""

import main


def TODO(message=""):
    raise NotImplementedError("TODO %s" % message)


def prob_to_energy(p):
    return log(1 / p - 1)


def energy_to_prob(e):
    return 1 / (1 + exp(e))


def helmholtz_free_energy(energies):
    return -log(sum(exp(-e) for e in energies))


# ----( factored matching )----------------------------------------------------

Port = int
Energy = float


class Vert:
    """
    Each vertex associates with either:
    (a) no edges, with energy F, or
    (b) for each port p, either:
      (1) no edge, with F-conditional energy M, or
      (2) exactly one edge, with F+M conditional probability A
    When self has only a single port, PM will always be zero.
    """

    def __init__(self):
        self.E = {}  # edges will add themselves

    def validate(self):
        for p, e in self.E.items():
            assert isinstance(p, Port)
            assert isinstance(e, Edge)
            assert self in e.V
            assert e.V[self] is p

    def validate_problem(self):
        assert isinstance(self.F, Energy)

        assert isinstance(self.M, dict)
        for p, energy in self.M.items():
            assert isinstance(p, Port)
            assert isinstance(e, Energy)

        assert isinstance(self.A, dict)
        for p, A in self.A.items():
            assert isinstance(p, Port)
            assert isinstance(A, dict)
            for e, energy in A.items():
                assert isinstance(e, Edge)
                assert isinstance(energy, Energy)

    def validate_soln(self):
        pass

    def init(self, F, M):
        "Set conditional energies."
        self.A = dict((p, dict((e, 0.0) for e in E)) for p, E in self.E.items())
        self.F = F
        self.M = M

    def false_alarm_energy(self):
        return self.F

    def missed_detection_energy(self):
        return self.M - self.F

    def assoc_energy(self, p, e):
        return self.A[p][e] - self.M - self.F

    def propagate(self):
        "Aggregate messages and normalize."
        for p, E in self.A.items():
            for e in E:
                E[e] = e.assoc_energy() - self.M[p] - self.F
        for p, E in self.A.items():
            shift = helmholtz_free_energy(iter(E.values()))
            for e in E:
                E[e] -= shift
            self.M[p] -= shift
            self.F -= shift

    def constrain(self):
        """
        Set assoc prob to max over ports, adding minimum mass to M.
        An alternative strategy would be to distribute mass between PF and PM.
        """
        self.PA = dict((p, dict((e, e.PA) for e in E)) for p, E in self.E.items())
        PM = dict((p, 1 - sum(PA)) for (p, PA) in self.PA.items())  # overestimated
        self.PF = min(PM.values())
        self.PM = dict((p, PMp - self.PF) for (p, PMp) in PM.items())  # minimum


class Edge:
    """
    Each edge associates with either:
    (a) all its vertices, with energy A, or
    (b) no vertices
    Edges may only associate with one port per vertex,
    to avoid double-counting messages.
    """

    def __init__(self, V):
        self.V = V
        for v, p in V.items():
            try:
                v.E[p].append(self)
            except KeyError:
                v.E[p] = [self]

    def validate(self):
        assert isinstance(self.V, dict)
        for v, p in self.V.items():
            assert isinstance(v, Vert)
            assert isinstance(p, Port)
            assert p in v.E
            assert self in v.E[p]

    def validate_problem(self):
        assert isinstance(self.A, Energy)

    def validate_soln(self, tol=1e-6):
        for v, p in self.V.items():
            assert abs(self.PA - v.PA[p][self]) < tol

    def init(self, A):
        self.A = A

    def assoc_energy(self):
        return self.A

    def nonassoc_energy(self):
        return -self.A

    def propagate(self):
        "aggregate messages"
        # Version 1.
        # A = 1.0
        # for v,p in self.V.iteritems():
        #  A += v.assoc_energy(p,self) - self.A
        # self.A = A
        # Version 2.
        self.A *= 1 - len(self.V)
        for v, p in self.V.items():
            self.A += v.assoc_energy(p, self)

    def constrain(self):
        "Set assoc prob to min over vertex views."
        self.PA = energy_to_prob(
            max(v.assoc_energy(p, self) for v, p in self.V.items())
        )


class Matching:
    """
    Given:
      V   vertices
      E   edges
      F   false alarm energies
      M   missed detection energies, relative to F
      A   association energies, relative to F,M
    Vary:
      e.A
      v.A
      v.F
      v.M
    To Minimize:
      "Kikuchi free energy"
    Subject To:
      "vertices agree"
    """

    def __init__(self, V, E, A, F, M):
        self.V = V
        self.E = E

        self.A = A
        self.F = F
        self.M = M

    def validate(self):

        # V : list Vert
        assert isinstance(self.V, list)
        for v in self.V:
            v.validate()
            for E in v.E.values():
                for e in E:
                    assert e in self.E

        # E : list Edge
        assert isinstance(self.E, list)
        for e in self.E:
            e.validate()
            for v in e.V.keys():
                assert v in self.V

        # A : E -> Energy
        assert isinstance(self.A, dict)
        for e, energy in self.A.items():
            assert e in self.E
            assert isinstance(energy, Energy)

        # F : V -> Energy
        assert isinstance(self.F, dict)
        for v, energy in self.F.items():
            assert v in self.V
            assert isinstance(energy, Energy)

        # M : V -> P -> Energy
        assert isinstance(self.M, dict)
        for v, M in self.M.items():
            assert v in self.V
            assert isinstance(M, dict)
            for p, energy in M.items():
                assert isinstance(p, Port)
                assert isinstance(energy, Energy)

    def solve(self, num_iters, validate=True):
        for e in self.E:
            e.init(self.A[e])
        for v in self.V:
            v.init(self.F[v], self.M[v])

        if validate:
            for e in self.E:
                e.validate_problem()
            for v in self.V:
                v.validate_problem()

        for v in self.V:
            v.propagate()
        for i in range(num_iters):
            for e in self.E:
                e.propagate()
            for v in self.V:
                v.propagate()

        for e in self.E:
            e.constrain()
        for v in self.V:
            v.constrain()

        if validate:
            for e in self.E:
                e.validate_soln()
            for v in self.V:
                v.validate_soln()

        self.PA = dict((e, e.PA) for e in self.E)
        self.PF = dict((v, v.PF) for v in self.V)
        self.PM = dict((v, v.PM) for v in self.V)


# ----( N-D assignment )-------------------------------------------------------


class MultiAssignment:
    """
    This does not fit elegantly into the framework.
    Let
      P = {1, ..., n}
      V = V1 + ... + Vn, where Vi.M[i] = -infty
      E (= V1 x ... x Vn
    """

    def __init__(self, frames, A, F):
        TODO("set M to +- infty")


# ----( factored N-D assignment )----------------------------------------------


class FactoredMultiAssignment:
    """
    Let
      P = {fwd = 1, bwd = 2}
      V = V1 + ... + Vn               # frames of detections
      E (= Union m<n. Vm x Vn         # pairwise associations
    """

    def __init__(self, frames, Pfa, Pa, Pmd):
        TODO("set up factored multi-assignment problem")


# ----( grid matching )--------------------------------------------------------


class GridMatching(Matching):
    """
    Let
      V = "gridpoints"
      E = Evert + Ehoriz
    where for each orientation i,
      Ei (= V x V
    """

    def __init__(self, points, Avertical, Ahoriz, F, Mup, Mdown, Mleft, Mright):
        P = list(range(4))
        self.verts = dict((p, Vertex()) for p in points)
        self.edges = {}
        for orient, Aorient in [("v", Avertical), ("h", Ahoriz)]:
            TODO("add edges")
        Matching.__init__(self, V, E, A, F, M)


@main.command
def test_grid(filename):
    "Weighs grid points and edges from point list file"
    TODO("read from calibration file")


# ----( main harness )---------------------------------------------------------


@main.command
def test():
    "Run all unit tests"
    test_grid("test.text")


if __name__ == "__main__":
    main.main()

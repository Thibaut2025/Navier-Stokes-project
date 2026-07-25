"""Elements finis : triangle de reference, fonctions de forme, matrices locales.

Ce module fait le lien entre la geometrie discrete produite par mesh.py et
les integrales que matrices.py devra assembler. Trois briques :

    1. Le triangle de reference K_hat = {(xi,eta) : xi>=0, eta>=0, xi+eta<=1}
       et les fonctions de forme dessus (P1 pour la pression et le maillage
       geometrique, P2 pour la vitesse et la temperature -- element de
       Taylor-Hood, cf. Doc.pdf : c'est le choix qui assure la condition
       inf-sup et evite les modes de pression parasites).
    2. Le changement de variables F_K : K_hat -> K (triangle physique), son
       jacobien et son determinant, qui permettent de transporter gradients
       et integrales.
    3. Les matrices elementaires de base (masse, rigidite) sur un triangle
       physique, brique de depart de l'assemblage global (module 4).

Convention : toutes les fonctions qui prennent "vertices" attendent un
tableau (3, 2) -- les 3 sommets du triangle, dans l'ordre CCW produit par
mesh.build_structured_triangular_mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh import Mesh2D


# ---------------------------------------------------------------------------
# 1. Le triangle de reference et ses noeuds
# ---------------------------------------------------------------------------

# P1 : un noeud par sommet.
REFERENCE_NODES_P1 = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=float,
)

# P2 : les 3 sommets + les 3 milieux d'aretes, dans l'ordre (v0,v1,v2,m01,m12,m20).
REFERENCE_NODES_P2 = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
    ],
    dtype=float,
)


def reference_p1_basis(xi: float, eta: float) -> np.ndarray:
    """Fonctions de forme P1 au point (xi, eta) du triangle de reference.

    En coordonnees barycentriques L1 = 1-xi-eta, L2 = xi, L3 = eta, les trois
    fonctions de forme P1 sont exactement L1, L2, L3 (affines, valant 1 sur
    leur propre sommet et 0 sur les deux autres).
    """

    return np.array([1.0 - xi - eta, xi, eta], dtype=float)


def reference_p1_gradients() -> np.ndarray:
    """Gradients (constants) des fonctions de forme P1.

    Returns
    -------
    np.ndarray
        Tableau (3, 2) : ligne i = [dphi_i/dxi, dphi_i/deta].
    """

    return np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)


def reference_p2_basis(xi: float, eta: float) -> np.ndarray:
    """Fonctions de forme P2 au point (xi, eta) du triangle de reference.

    Construites a partir des coordonnees barycentriques L1,L2,L3 : les 3
    premieres (sommets) valent L_k*(2*L_k-1), les 3 suivantes (milieux
    d'aretes) valent 4*L_i*L_j pour l'arete (i,j).
    """

    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta

    return np.array(
        [
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ],
        dtype=float,
    )


def reference_p2_gradients(xi: float, eta: float) -> np.ndarray:
    """Gradients des fonctions de forme P2 au point (xi, eta).

    Returns
    -------
    np.ndarray
        Tableau (6, 2) : ligne i = [dphi_i/dxi, dphi_i/deta].
    """

    return np.array(
        [
            [4.0 * xi + 4.0 * eta - 3.0, 4.0 * xi + 4.0 * eta - 3.0],
            [4.0 * xi - 1.0, 0.0],
            [0.0, 4.0 * eta - 1.0],
            [4.0 - 8.0 * xi - 4.0 * eta, -4.0 * xi],
            [4.0 * eta, 4.0 * xi],
            [-4.0 * eta, 4.0 - 4.0 * xi - 8.0 * eta],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 2. Changement de variables triangle de reference -> triangle physique
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriangleGeometry:
    """Geometrie d'un triangle physique K, image de K_hat par F_K.

    Attributes
    ----------
    vertices:
        Tableau (3, 2) des sommets physiques.
    jacobian:
        Jacobien J de F_K (constant sur K car F_K est affine en P1).
    determinant:
        det(J). Son signe indique l'orientation (positif = CCW, comme
        impose par mesh.py) ; sa valeur absolue relie les aires :
        aire(K) = |det(J)| * aire(K_hat) = |det(J)| / 2.
    inverse_jacobian:
        J^{-1}, necessaire pour transporter les gradients (regle de la
        chaine : grad_x phi = J^{-T} grad_hat phi).
    area:
        Aire du triangle physique.
    """

    vertices: np.ndarray
    jacobian: np.ndarray
    determinant: float
    inverse_jacobian: np.ndarray
    area: float


def triangle_geometry(vertices: np.ndarray) -> TriangleGeometry:
    """Calcule le jacobien, son determinant et l'aire d'un triangle physique.

    Parameters
    ----------
    vertices:
        Tableau (3, 2) : sommets (v0, v1, v2) du triangle physique.
    """

    if vertices.shape != (3, 2):
        raise ValueError("vertices doit avoir la forme (3, 2).")

    v0, v1, v2 = vertices[0], vertices[1], vertices[2]

    jacobian = np.array(
        [
            [v1[0] - v0[0], v2[0] - v0[0]],
            [v1[1] - v0[1], v2[1] - v0[1]],
        ],
        dtype=float,
    )

    determinant = float(np.linalg.det(jacobian))
    if abs(determinant) < 1e-14:
        raise ValueError("Triangle degenere : determinant du jacobien nul.")

    inverse_jacobian = np.linalg.inv(jacobian)
    area = 0.5 * abs(determinant)

    return TriangleGeometry(
        vertices=vertices,
        jacobian=jacobian,
        determinant=determinant,
        inverse_jacobian=inverse_jacobian,
        area=area,
    )


def physical_gradients(reference_gradients: np.ndarray, inverse_jacobian: np.ndarray) -> np.ndarray:
    """Transporte des gradients du triangle de reference vers le triangle physique.

    Regle de la chaine pour F_K affine : grad_x phi = J^{-T} grad_hat phi,
    appliquee ici a toutes les fonctions de forme a la fois.
    """

    return reference_gradients @ inverse_jacobian.T


# ---------------------------------------------------------------------------
# 3. Quadrature sur le triangle de reference
# ---------------------------------------------------------------------------
#
# Regle de Dunavant a 7 points, exacte pour les polynomes de degre <= 5.
# Necessaire pour integrer les produits de fonctions P2 (degre 4 dans la
# masse) et les termes de convection (degre impair a cause de u.grad) qu'on
# rencontrera aux modules 4 et 6-7 : une formule fermee comme pour P1 n'est
# plus pratique, on integre donc numeriquement partout ou P2 intervient.

def reference_triangle_quadrature() -> tuple[np.ndarray, np.ndarray]:
    """Points et poids de la quadrature de Dunavant d'ordre 5.

    Les poids sont deja mis a l'echelle de l'aire du triangle de reference
    (1/2) : `sum(weights) == 0.5`.
    """

    points = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.059715871789770, 0.470142064105115],
            [0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087],
            [0.797426985353087, 0.101286507323456],
        ],
        dtype=float,
    )

    weights = 0.5 * np.array(
        [
            0.225000000000000,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ],
        dtype=float,
    )

    return points, weights


# ---------------------------------------------------------------------------
# 4. Matrices locales de base (point de depart de l'assemblage, module 4)
# ---------------------------------------------------------------------------

def local_mass_matrix_p1(vertices: np.ndarray) -> np.ndarray:
    """Matrice locale de masse P1 : int_K phi_i phi_j dx.

    Formule fermee classique (le produit de deux fonctions P1 est degre 2,
    integrable exactement sans quadrature).
    """

    area = triangle_geometry(vertices).area
    return (area / 12.0) * np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=float,
    )


def local_stiffness_matrix_p1(vertices: np.ndarray) -> np.ndarray:
    """Matrice locale de rigidite P1 : int_K grad(phi_i).grad(phi_j) dx.

    Les gradients P1 sont constants sur K, donc l'integrale est juste le
    produit scalaire des gradients fois l'aire.
    """

    geometry = triangle_geometry(vertices)
    gradients = physical_gradients(reference_p1_gradients(), geometry.inverse_jacobian)
    return geometry.area * (gradients @ gradients.T)


def local_mass_matrix_p2(vertices: np.ndarray) -> np.ndarray:
    """Matrice locale de masse P2 : int_K phi_i phi_j dx, par quadrature."""

    geometry = triangle_geometry(vertices)
    points, weights = reference_triangle_quadrature()

    mass = np.zeros((6, 6), dtype=float)
    for (xi, eta), weight in zip(points, weights, strict=True):
        basis = reference_p2_basis(float(xi), float(eta))
        mass += weight * np.outer(basis, basis)

    return abs(geometry.determinant) * mass


def local_stiffness_matrix_p2(vertices: np.ndarray) -> np.ndarray:
    """Matrice locale de rigidite P2 : int_K grad(phi_i).grad(phi_j) dx, par quadrature."""

    geometry = triangle_geometry(vertices)
    points, weights = reference_triangle_quadrature()

    stiffness = np.zeros((6, 6), dtype=float)
    for (xi, eta), weight in zip(points, weights, strict=True):
        gradients = physical_gradients(reference_p2_gradients(float(xi), float(eta)), geometry.inverse_jacobian)
        stiffness += weight * (gradients @ gradients.T)

    return abs(geometry.determinant) * stiffness


# ---------------------------------------------------------------------------
# 5. Enrichissement P1 -> P2 : ajouter les noeuds milieux d'aretes au maillage
# ---------------------------------------------------------------------------
#
# La vitesse et la temperature vivent sur les degres de liberte P2 (element
# de Taylor-Hood : vitesse P2, pression P1, cf. notes.txt etape 4). Il faut
# donc un maillage enrichi ou chaque arete du maillage geometrique porte un
# degre de liberte supplementaire, partage entre les deux triangles qui
# bordent l'arete.

@dataclass(frozen=True)
class QuadraticMesh2D:
    """Maillage P2 obtenu en enrichissant un Mesh2D (P1) de mesh.py.

    Attributes
    ----------
    nodes:
        Tableau (n_nodes, 2) : les noeuds P1 d'origine suivis des noeuds
        milieux d'aretes ajoutes.
    triangles:
        Connectivite (n_elements, 6), ordre (v0, v1, v2, m01, m12, m20),
        coherent avec REFERENCE_NODES_P2.
    edge_to_midpoint:
        Dictionnaire {(noeud_a, noeud_b) trie : indice du noeud milieu},
        pour retrouver rapidement un degre de liberte d'arete.
    """

    nodes: np.ndarray
    triangles: np.ndarray
    edge_to_midpoint: dict[tuple[int, int], int]

    @property
    def n_nodes(self) -> int:
        """Nombre total de noeuds P2 (sommets + milieux d'aretes)."""

        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        """Nombre total de triangles (identique au maillage P1 sous-jacent)."""

        return int(self.triangles.shape[0])


def build_quadratic_mesh(mesh: Mesh2D) -> QuadraticMesh2D:
    """Enrichit un maillage P1 en maillage P2 en ajoutant les milieux d'aretes.

    Chaque arete est partagee par au plus deux triangles ; le dictionnaire
    edge_to_midpoint garantit qu'on ne cree qu'un seul noeud milieu par
    arete, meme quand on la rencontre depuis ses deux triangles voisins.
    """

    nodes = list(mesh.nodes)
    edge_to_midpoint: dict[tuple[int, int], int] = {}
    quadratic_triangles = np.zeros((mesh.n_elements, 6), dtype=int)

    def midpoint_index(node_a: int, node_b: int) -> int:
        edge = (min(node_a, node_b), max(node_a, node_b))
        midpoint = edge_to_midpoint.get(edge)
        if midpoint is None:
            midpoint = len(nodes)
            nodes.append(0.5 * (mesh.nodes[edge[0]] + mesh.nodes[edge[1]]))
            edge_to_midpoint[edge] = midpoint
        return midpoint

    for element_index, triangle in enumerate(mesh.triangles):
        v0, v1, v2 = (int(v) for v in triangle)
        m01 = midpoint_index(v0, v1)
        m12 = midpoint_index(v1, v2)
        m20 = midpoint_index(v2, v0)
        quadratic_triangles[element_index, :] = (v0, v1, v2, m01, m12, m20)

    return QuadraticMesh2D(
        nodes=np.asarray(nodes, dtype=float),
        triangles=quadratic_triangles,
        edge_to_midpoint=edge_to_midpoint,
    )
"""Assemblage global des matrices elements finis (etape 4).

Le module precedent (finite_elements.py) sait calculer, sur UN triangle, les
petites matrices locales (3x3 en P1, 6x6 en P2). Ce module fait le pas
suivant : parcourir tous les triangles du maillage et deposer chaque
contribution locale a la bonne place dans une grande matrice globale creuse.

C'est le coeur mecanique de la methode des elements finis : une integrale sur
Omega est une somme d'integrales sur les triangles, donc la matrice globale
est la somme des matrices locales "dispatchees" par la table de connectivite.

Matrices produites, et a quoi elles servent dans la suite :

    M_p1, K_p1   masse et rigidite sur les noeuds sommets  -> pression
    M_p2, K_p2   masse et rigidite sur les noeuds P2        -> vitesse, temperature
    B            couplage vitesse-pression (divergence)     -> blocs hors-diagonale
                 du systeme de Stokes (Doc.pdf, forme matricielle eq. 14)
    N(u_h)       convection linearisee, dependante du champ de vitesse courant
                 -> reassemblee a chaque pas de temps (module 9)

Format : scipy.sparse. On accumule les contributions en triplets (i, j, v)
puis on convertit en CSR ; c'est la facon standard de faire, et le passage en
CSR somme automatiquement les doublons (un meme couple (i,j) recoit une
contribution de chaque triangle partage).

Convention pour les degres de liberte de vitesse
------------------------------------------------
La vitesse a deux composantes, chacune discretisee en P2. Le vecteur global
de vitesse est range par composante :

    v = [ vx_0 ... vx_{n-1} | vy_0 ... vy_{n-1} ]   (taille 2 * n_nodes_p2)

C'est ce rangement "par blocs" qui rend le systeme de Stokes lisible : les
blocs de diffusion sont alors deux copies de K_p2 sur la diagonale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from finite_elements import (
    QuadraticMesh2D,
    build_quadratic_mesh,
    local_mass_matrix_p1,
    local_mass_matrix_p2,
    local_stiffness_matrix_p1,
    local_stiffness_matrix_p2,
    physical_gradients,
    reference_p1_basis,
    reference_p2_basis,
    reference_p2_gradients,
    reference_triangle_quadrature,
    triangle_geometry,
)
from mesh import Mesh2D, build_structured_triangular_mesh


# ---------------------------------------------------------------------------
# 1. Brique d'assemblage : dispatcher une matrice locale dans la globale
# ---------------------------------------------------------------------------

def _scatter(
    local: np.ndarray,
    row_dofs: np.ndarray,
    col_dofs: np.ndarray,
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    values: list[np.ndarray],
) -> None:
    """Ajoute une matrice locale a la liste des triplets (i, j, valeur).

    ``local[a, b]`` est la contribution du couple de fonctions de forme
    (a, b) LOCALES ; elle doit atterrir en ligne ``row_dofs[a]`` et colonne
    ``col_dofs[b]`` de la matrice globale. On ne somme rien ici : les
    doublons seront additionnes lors de la conversion en CSR.
    """

    rows.append(np.repeat(row_dofs, col_dofs.size))
    cols.append(np.tile(col_dofs, row_dofs.size))
    values.append(local.ravel())


def _to_csr(
    rows: list[np.ndarray],
    cols: list[np.ndarray],
    values: list[np.ndarray],
    shape: tuple[int, int],
) -> sp.csr_matrix:
    """Assemble les triplets accumules en une matrice CSR.

    Le passage par ``coo_matrix`` est volontaire : c'est lui qui additionne
    les contributions des differents triangles sur un meme couple (i, j).
    """

    matrix = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
    )
    return matrix.tocsr()


# ---------------------------------------------------------------------------
# 2. Matrices scalaires : masse et rigidite, en P1 et en P2
# ---------------------------------------------------------------------------

def assemble_mass_p1(mesh: Mesh2D) -> sp.csr_matrix:
    """Matrice de masse P1 globale : M[i,j] = int_Omega phi_i phi_j dx."""

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for triangle in mesh.triangles:
        local = local_mass_matrix_p1(mesh.nodes[triangle])
        _scatter(local, triangle, triangle, rows, cols, values)

    return _to_csr(rows, cols, values, (mesh.n_nodes, mesh.n_nodes))


def assemble_stiffness_p1(mesh: Mesh2D) -> sp.csr_matrix:
    """Matrice de rigidite P1 globale : K[i,j] = int_Omega grad(phi_i).grad(phi_j) dx."""

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for triangle in mesh.triangles:
        local = local_stiffness_matrix_p1(mesh.nodes[triangle])
        _scatter(local, triangle, triangle, rows, cols, values)

    return _to_csr(rows, cols, values, (mesh.n_nodes, mesh.n_nodes))


def assemble_mass_p2(quadratic: QuadraticMesh2D) -> sp.csr_matrix:
    """Matrice de masse P2 globale, sur les noeuds sommets + milieux d'aretes."""

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for triangle in quadratic.triangles:
        # La geometrie reste affine : seuls les 3 sommets definissent le triangle.
        local = local_mass_matrix_p2(quadratic.nodes[triangle[:3]])
        _scatter(local, triangle, triangle, rows, cols, values)

    return _to_csr(rows, cols, values, (quadratic.n_nodes, quadratic.n_nodes))


def assemble_stiffness_p2(quadratic: QuadraticMesh2D) -> sp.csr_matrix:
    """Matrice de rigidite P2 globale (diffusion de la vitesse et de la temperature)."""

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for triangle in quadratic.triangles:
        local = local_stiffness_matrix_p2(quadratic.nodes[triangle[:3]])
        _scatter(local, triangle, triangle, rows, cols, values)

    return _to_csr(rows, cols, values, (quadratic.n_nodes, quadratic.n_nodes))


# ---------------------------------------------------------------------------
# 3. Couplage vitesse-pression : la matrice de divergence B
# ---------------------------------------------------------------------------
#
# Dans la formulation faible de Stokes, le terme de pression apparait sous la
# forme int_Omega q * div(v) dx, ou q est une fonction test de pression (P1)
# et v une fonction test de vitesse (P2, 2 composantes). C'est ce que B
# discretise :
#
#     B[k, j]         = int_Omega phi1_k * d(phi2_j)/dx   (bloc composante x)
#     B[k, n_p2 + j]  = int_Omega phi1_k * d(phi2_j)/dy   (bloc composante y)
#
# et alors B @ v vaut exactement le vecteur des int q_k div(v_h). Sa
# transposee -B^T fournit le terme de gradient de pression dans l'equation de
# quantite de mouvement : c'est la structure en point-selle de Taylor-Hood.

def local_divergence_matrix(vertices: np.ndarray) -> np.ndarray:
    """Matrice locale de divergence, de forme (3, 6, 2).

    L'entree ``[k, j, d]`` vaut int_K phi1_k * d(phi2_j)/dx_d, avec d = 0
    pour x et d = 1 pour y. L'integrande est de degre 1 (P1) + 1 (gradient
    de P2) = 2 : la quadrature d'ordre 5 l'integre exactement.
    """

    geometry = triangle_geometry(vertices)
    points, weights = reference_triangle_quadrature()

    local = np.zeros((3, 6, 2), dtype=float)
    for (xi, eta), weight in zip(points, weights, strict=True):
        basis_p1 = reference_p1_basis(float(xi), float(eta))
        gradients_p2 = physical_gradients(
            reference_p2_gradients(float(xi), float(eta)), geometry.inverse_jacobian
        )
        local += weight * np.einsum("k,jd->kjd", basis_p1, gradients_p2)

    return abs(geometry.determinant) * local


def assemble_divergence(mesh: Mesh2D, quadratic: QuadraticMesh2D) -> sp.csr_matrix:
    """Matrice de divergence globale B, de forme (n_p1, 2 * n_p2).

    Les colonnes suivent la convention de rangement de la vitesse decrite en
    tete de module : d'abord les n_p2 degres de liberte de vx, puis ceux de vy.
    """

    n_p2 = quadratic.n_nodes

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for element_index, triangle in enumerate(quadratic.triangles):
        local = local_divergence_matrix(quadratic.nodes[triangle[:3]])
        pressure_dofs = mesh.triangles[element_index]

        # Les degres de liberte de vitesse d'une composante sont ceux du
        # maillage P2 ; le decalage n_p2 selectionne la composante y.
        _scatter(local[:, :, 0], pressure_dofs, triangle, rows, cols, values)
        _scatter(local[:, :, 1], pressure_dofs, triangle + n_p2, rows, cols, values)

    return _to_csr(rows, cols, values, (mesh.n_nodes, 2 * n_p2))


# ---------------------------------------------------------------------------
# 4. Convection : N(u_h), la seule matrice qui depend de la solution
# ---------------------------------------------------------------------------
#
# Le terme non lineaire (u.grad)u de Navier-Stokes, et le terme de transport
# u.grad(T) de l'equation de la chaleur, se discretisent tous les deux par le
# meme operateur, une fois le champ de vitesse u_h considere comme connu
# (linearisation semi-implicite : on prend u_h au pas de temps precedent) :
#
#     N[i, j] = int_Omega (u_h . grad(phi_j)) * phi_i dx
#
# C'est pourquoi cette matrice doit etre reassemblee a chaque pas de temps,
# contrairement a toutes les precedentes qui sont calculees une fois pour
# toutes. Elle n'est pas symetrique : le transport a un sens.

def local_convection_matrix(vertices: np.ndarray, element_velocity: np.ndarray) -> np.ndarray:
    """Matrice locale de convection (6, 6) pour une vitesse donnee sur l'element.

    Parameters
    ----------
    vertices:
        Tableau (3, 2) des sommets du triangle (la geometrie reste affine).
    element_velocity:
        Tableau (6, 2) : les deux composantes de u_h aux 6 noeuds P2 de
        l'element. La vitesse au point de quadrature est reconstruite par les
        fonctions de forme P2, comme n'importe quel champ discret.

    L'integrande est de degre 2 (u_h en P2) + 1 (gradient de P2) + 2 (phi_i
    en P2) = 5 : la quadrature de Dunavant d'ordre 5 deja disponible dans
    finite_elements.py l'integre donc exactement, sans rien ajouter.
    """

    geometry = triangle_geometry(vertices)
    points, weights = reference_triangle_quadrature()

    local = np.zeros((6, 6), dtype=float)
    for (xi, eta), weight in zip(points, weights, strict=True):
        basis = reference_p2_basis(float(xi), float(eta))
        gradients = physical_gradients(
            reference_p2_gradients(float(xi), float(eta)), geometry.inverse_jacobian
        )
        velocity_at_point = basis @ element_velocity      # (2,)
        transport = gradients @ velocity_at_point         # (6,) : u_h . grad(phi_j)
        local += weight * np.outer(basis, transport)

    return abs(geometry.determinant) * local


def assemble_convection(quadratic: QuadraticMesh2D, velocity: np.ndarray) -> sp.csr_matrix:
    """Matrice de convection globale N(u_h), de forme (n_p2, n_p2).

    Parameters
    ----------
    velocity:
        Champ de vitesse discret, soit de forme (n_p2, 2), soit aplati en
        (2 * n_p2,) selon la convention de rangement par composante decrite
        en tete de module.
    """

    n_p2 = quadratic.n_nodes
    if velocity.shape == (2 * n_p2,):
        velocity = np.column_stack((velocity[:n_p2], velocity[n_p2:]))
    if velocity.shape != (n_p2, 2):
        raise ValueError("velocity doit avoir la forme (n_p2, 2) ou (2 * n_p2,).")

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []

    for triangle in quadratic.triangles:
        local = local_convection_matrix(quadratic.nodes[triangle[:3]], velocity[triangle])
        _scatter(local, triangle, triangle, rows, cols, values)

    return _to_csr(rows, cols, values, (n_p2, n_p2))


# ---------------------------------------------------------------------------
# 5. Regroupement : tout ce qui est assemble une fois pour toutes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlobalMatrices:
    """Les matrices constantes du probleme, assemblees une seule fois.

    N(u_h) n'en fait volontairement pas partie : elle depend de la solution
    et sera recalculee a chaque pas de temps par ``assemble_convection``.

    Attributes
    ----------
    mesh, quadratic:
        Les deux maillages (P1 geometrique, P2 enrichi) sur lesquels tout a
        ete assemble ; conserves pour que les modules suivants n'aient pas a
        les reconstruire.
    mass_p1, stiffness_p1:
        Masse et rigidite sur les noeuds sommets (espace de la pression).
    mass_p2, stiffness_p2:
        Masse et rigidite sur les noeuds P2 (vitesse et temperature).
    divergence:
        Couplage vitesse-pression B, de forme (n_p1, 2 * n_p2).
    """

    mesh: Mesh2D
    quadratic: QuadraticMesh2D
    mass_p1: sp.csr_matrix
    stiffness_p1: sp.csr_matrix
    mass_p2: sp.csr_matrix
    stiffness_p2: sp.csr_matrix
    divergence: sp.csr_matrix

    @property
    def n_pressure_dofs(self) -> int:
        """Nombre de degres de liberte de pression (P1)."""

        return self.mesh.n_nodes

    @property
    def n_scalar_dofs(self) -> int:
        """Nombre de degres de liberte P2 (une composante de vitesse, ou T)."""

        return self.quadratic.n_nodes

    @property
    def n_velocity_dofs(self) -> int:
        """Nombre total de degres de liberte de vitesse (2 composantes P2)."""

        return 2 * self.quadratic.n_nodes


def assemble_all(mesh: Mesh2D | None = None) -> GlobalMatrices:
    """Assemble toutes les matrices constantes sur le maillage donne.

    Sans argument, construit le maillage par defaut de parameters.py.
    """

    if mesh is None:
        mesh = build_structured_triangular_mesh()
    quadratic = build_quadratic_mesh(mesh)

    return GlobalMatrices(
        mesh=mesh,
        quadratic=quadratic,
        mass_p1=assemble_mass_p1(mesh),
        stiffness_p1=assemble_stiffness_p1(mesh),
        mass_p2=assemble_mass_p2(quadratic),
        stiffness_p2=assemble_stiffness_p2(quadratic),
        divergence=assemble_divergence(mesh, quadratic),
    )

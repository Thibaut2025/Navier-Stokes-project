"""Construction du maillage 2D.

Ce module transforme le rectangle continu Omega (defini dans parameters.py)
en un maillage triangulaire discret : un ensemble fini de noeuds relies par
des triangles. C'est le squelette geometrique sur lequel les modules
suivants (finite_elements.py, matrices.py) vont poser les fonctions de
forme et assembler les matrices.

Strategie retenue : maillage structure. On decoupe Omega en nx*ny cellules
rectangulaires identiques, puis chaque cellule est coupee en 2 triangles par
sa diagonale. C'est le maillage le plus simple qui existe -- suffisant pour
un domaine rectangulaire comme le notre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parameters import DOMAIN, MESH_NX, MESH_NY


@dataclass(frozen=True)
class Mesh2D:
    """Maillage triangulaire 2D structure.

    Attributes
    ----------
    nodes:
        Tableau ``(n_nodes, 2)`` : coordonnees ``(x, y)`` de chaque noeud.
    triangles:
        Tableau ``(n_elements, 3)`` : pour chaque triangle, les indices (dans
        ``nodes``) de ses 3 sommets, ranges dans le sens trigonometrique.
    nx, ny:
        Nombre de subdivisions de la grille selon x et y.
    dx, dy:
        Pas de la grille (taille d'une cellule) selon x et y.
    boundary_nodes:
        Indices des noeuds situes sur chacun des 4 bords du rectangle.
        Utile pour le module 5 (conditions de Dirichlet u=0, T=0 sur
        partial Omega).
    """

    nodes: np.ndarray
    triangles: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    boundary_nodes: dict[str, np.ndarray]

    @property
    def n_nodes(self) -> int:
        """Nombre total de noeuds du maillage."""

        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        """Nombre total de triangles du maillage."""

        return int(self.triangles.shape[0])


def node_index(i: int, j: int, nx: int) -> int:
    """Convertit un indice de grille ``(i, j)`` en indice global de noeud.

    Meme logique qu'un ``numpy.ravel(order="C")`` sur un tableau
    ``(ny+1, nx+1)`` : on parcourt une ligne (x croissant) avant de passer a
    la ligne suivante (y croissant).
    """

    return j * (nx + 1) + i


def build_structured_triangular_mesh(nx: int = MESH_NX, ny: int = MESH_NY) -> Mesh2D:
    """Construit le maillage triangulaire structure sur le domaine ``DOMAIN``.

    Parameters
    ----------
    nx, ny:
        Nombre de subdivisions de la grille selon x et y.

    Returns
    -------
    Mesh2D
        Le maillage complet : noeuds, triangles, et noeuds de bord.
    """

    if nx <= 0 or ny <= 0:
        raise ValueError("nx et ny doivent etre strictement positifs.")

    dx = DOMAIN.length_x / nx
    dy = DOMAIN.length_y / ny

    # --- 1. Les noeuds : une grille reguliere de (nx+1) x (ny+1) points ---
    n_nodes = (nx + 1) * (ny + 1)
    nodes = np.zeros((n_nodes, 2), dtype=float)
    for j in range(ny + 1):
        y_coordinate = DOMAIN.y_min + j * dy
        for i in range(nx + 1):
            x_coordinate = DOMAIN.x_min + i * dx
            nodes[node_index(i, j, nx), :] = (x_coordinate, y_coordinate)

    # --- 2. Les triangles : chaque cellule (i,j) -> 2 triangles CCW ---
    n_elements = 2 * nx * ny
    triangles = np.zeros((n_elements, 3), dtype=int)
    element_counter = 0
    for j in range(ny):
        for i in range(nx):
            n00 = node_index(i, j, nx)
            n10 = node_index(i + 1, j, nx)
            n01 = node_index(i, j + 1, nx)
            n11 = node_index(i + 1, j + 1, nx)

            # Diagonale n00-n11 : deux triangles dans le sens trigonometrique.
            triangles[element_counter, :] = (n00, n10, n11)
            element_counter += 1
            triangles[element_counter, :] = (n00, n11, n01)
            element_counter += 1

    # --- 3. Les noeuds de bord : necessaires pour les conditions de Dirichlet ---
    boundary_nodes = {
        "left": np.array([node_index(0, j, nx) for j in range(ny + 1)], dtype=int),
        "right": np.array([node_index(nx, j, nx) for j in range(ny + 1)], dtype=int),
        "bottom": np.array([node_index(i, 0, nx) for i in range(nx + 1)], dtype=int),
        "top": np.array([node_index(i, ny, nx) for i in range(nx + 1)], dtype=int),
    }

    return Mesh2D(
        nodes=nodes,
        triangles=triangles,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        boundary_nodes=boundary_nodes,
    )


def all_boundary_dofs(mesh: Mesh2D) -> np.ndarray:
    """Retourne, sans doublons, tous les indices de noeuds situes sur le bord.

    Les 4 cotes partagent les 4 coins du rectangle : on doit dedupliquer.
    """

    return np.unique(
        np.concatenate(
            [
                mesh.boundary_nodes["left"],
                mesh.boundary_nodes["right"],
                mesh.boundary_nodes["bottom"],
                mesh.boundary_nodes["top"],
            ]
        )
    )

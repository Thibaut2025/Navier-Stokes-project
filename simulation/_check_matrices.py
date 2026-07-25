"""Script de verification du module 4 (matrices.py).

Test manuel : assembler toutes les matrices globales et verifier, par des
proprietes mathematiques connues, qu'elles sont correctes avant de passer aux
conditions aux limites (module 5).

Les proprietes testees ici ne dependent d'aucune solution de reference : ce
sont des identites que TOUTE matrice elements finis correctement assemblee
doit verifier. C'est ce qui en fait un bon filet de securite.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import parameters as p
from matrices import assemble_all, assemble_convection


def symmetry_error(matrix: sp.csr_matrix) -> float:
    """Ecart maximal entre la matrice et sa transposee."""

    return float(abs(matrix - matrix.T).max())


def main() -> None:
    matrices = assemble_all()
    mesh = matrices.mesh
    quadratic = matrices.quadratic

    n_p1 = matrices.n_pressure_dofs
    n_p2 = matrices.n_scalar_dofs

    print("=== Degres de liberte ===")
    print(f"P1 (pression)            : {n_p1}")
    print(f"P2 (vitesse / temperature) : {n_p2}")
    print(f"vitesse (2 composantes)  : {matrices.n_velocity_dofs}")

    # Le maillage P2 ajoute un noeud par arete. Pour une triangulation plane,
    # la relation d'Euler donne aretes = sommets + triangles - 1.
    expected_edges = mesh.n_nodes + mesh.n_elements - 1
    print(f"\nNoeuds P2 attendus : {mesh.n_nodes} sommets + {expected_edges} aretes = {mesh.n_nodes + expected_edges}")
    assert n_p2 == mesh.n_nodes + expected_edges, "Le compte de noeuds P2 ne colle pas a la relation d'Euler."

    print("\n=== Formes des matrices ===")
    for name, matrix, shape in [
        ("mass_p1", matrices.mass_p1, (n_p1, n_p1)),
        ("stiffness_p1", matrices.stiffness_p1, (n_p1, n_p1)),
        ("mass_p2", matrices.mass_p2, (n_p2, n_p2)),
        ("stiffness_p2", matrices.stiffness_p2, (n_p2, n_p2)),
        ("divergence", matrices.divergence, (n_p1, 2 * n_p2)),
    ]:
        density = 100.0 * matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        print(f"{name:14s} {str(matrix.shape):>16s}  nnz = {matrix.nnz:>7d}  ({density:.2f} % de remplissage)")
        assert matrix.shape == shape, f"Forme inattendue pour {name}."

    print("\n=== Symetrie (masse et rigidite doivent etre symetriques) ===")
    for name, matrix in [
        ("mass_p1", matrices.mass_p1),
        ("stiffness_p1", matrices.stiffness_p1),
        ("mass_p2", matrices.mass_p2),
        ("stiffness_p2", matrices.stiffness_p2),
    ]:
        error = symmetry_error(matrix)
        print(f"{name:14s} : ecart max a la transposee = {error:.3e}")
        assert error < 1e-12, f"{name} n'est pas symetrique."

    # Somme de tous les coefficients de la masse : les fonctions de forme
    # forment une partition de l'unite (sum_i phi_i = 1), donc
    # sum_ij M_ij = int_Omega 1 * 1 dx = aire du domaine.
    print("\n=== Masse : partition de l'unite (somme = aire du domaine) ===")
    area = p.DOMAIN.length_x * p.DOMAIN.length_y
    print(f"aire exacte du domaine : {area:.12f}")
    for name, matrix in [("mass_p1", matrices.mass_p1), ("mass_p2", matrices.mass_p2)]:
        total = float(matrix.sum())
        print(f"{name:14s} : somme des coefficients = {total:.12f}  (erreur {abs(total - area):.3e})")
        assert abs(total - area) < 1e-12, f"La somme de {name} ne vaut pas l'aire du domaine."

    # Un champ constant a un gradient nul : K @ 1 = 0. C'est le test qui
    # detecte le plus surement une erreur de signe ou de jacobien.
    print("\n=== Rigidite : un champ constant a une energie nulle (K @ 1 = 0) ===")
    for name, matrix, size in [
        ("stiffness_p1", matrices.stiffness_p1, n_p1),
        ("stiffness_p2", matrices.stiffness_p2, n_p2),
    ]:
        residual = float(np.abs(matrix @ np.ones(size)).max())
        print(f"{name:14s} : max |K @ 1| = {residual:.3e}")
        assert residual < 1e-10, f"{name} ne s'annule pas sur un champ constant."

    # Un champ de vitesse uniforme est a divergence nulle : B @ v = 0.
    print("\n=== Divergence : une vitesse uniforme est incompressible (B @ v = 0) ===")
    uniform = np.zeros(2 * n_p2)
    uniform[:n_p2] = 1.0        # vx = 1 partout
    uniform[n_p2:] = -0.5       # vy = -0.5 partout
    residual = float(np.abs(matrices.divergence @ uniform).max())
    print(f"max |B @ v_uniforme| = {residual:.3e}")
    assert residual < 1e-10, "La matrice de divergence ne s'annule pas sur un champ uniforme."

    # La convection transporte un champ : appliquee a un champ constant, elle
    # rend zero (grad d'une constante = 0), quel que soit le champ de vitesse.
    print("\n=== Convection : N(u) @ 1 = 0 pour toute vitesse u ===")
    rng = np.random.default_rng(0)
    random_velocity = rng.normal(size=(n_p2, 2))
    convection = assemble_convection(quadratic, random_velocity)
    residual = float(np.abs(convection @ np.ones(n_p2)).max())
    print(f"forme N(u) : {convection.shape},  nnz = {convection.nnz}")
    print(f"max |N(u) @ 1| = {residual:.3e}")
    assert residual < 1e-10, "La matrice de convection ne s'annule pas sur un champ constant."

    # La convection n'est pas symetrique : le transport a un sens. On le
    # verifie explicitement pour eviter de croire a une erreur plus tard.
    asymmetry = symmetry_error(convection)
    print(f"ecart a la transposee = {asymmetry:.3e}  (doit etre non nul : le transport est oriente)")
    assert asymmetry > 1e-6, "N(u) est symetrique : le terme de transport a probablement ete perdu."

    # --- Visuel : les motifs de remplissage (structure creuse) ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))

    for ax, (title, matrix) in zip(
        axes,
        [
            ("masse P1 (pression)", matrices.mass_p1),
            ("rigidite P2 (vitesse, T)", matrices.stiffness_p2),
            ("divergence B (couplage u-p)", matrices.divergence),
            ("convection N(u)", convection),
        ],
    ):
        ax.spy(matrix, markersize=0.25, color="#1f77b4")
        ax.set_title(f"{title}\n{matrix.shape[0]}x{matrix.shape[1]}, nnz = {matrix.nnz}", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Motifs de remplissage des matrices globales (maillage {mesh.nx}x{mesh.ny})",
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    output_path = "figures/03_matrices_check.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"\nFigure enregistree : {output_path}")
    print("\nToutes les verifications sont passees.")


if __name__ == "__main__":
    main()

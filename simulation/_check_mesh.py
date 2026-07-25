"""Script de verification du module 2 (mesh.py).

Test manuel : construire le maillage et verifier, a l'oeil et par quelques
chiffres, qu'il est correct avant de passer au module suivant
(finite_elements.py).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import parameters as p
from mesh import all_boundary_dofs, build_structured_triangular_mesh


def main() -> None:
    mesh = build_structured_triangular_mesh()

    print("=== Dimensions du maillage ===")
    print(f"n_nodes    = {mesh.n_nodes}  (attendu : {(mesh.nx + 1) * (mesh.ny + 1)})")
    print(f"n_elements = {mesh.n_elements}  (attendu : {2 * mesh.nx * mesh.ny})")
    print(f"dx = {mesh.dx:.4f}, dy = {mesh.dy:.4f}")

    print("\n=== Noeuds de bord ===")
    for side, indices in mesh.boundary_nodes.items():
        print(f"{side:8s} : {indices.size} noeuds")
    boundary = all_boundary_dofs(mesh)
    print(f"total (sans doublons aux coins) : {boundary.size}")
    attendu = 2 * (mesh.nx + mesh.ny)
    print(f"attendu (perimetre) : {attendu}")
    assert boundary.size == attendu, "Le nombre de noeuds de bord ne correspond pas au perimetre attendu."

    print("\n=== Verification d'orientation (aire signee) ===")
    # Une aire signee positive confirme l'ordre CCW des sommets de chaque triangle.
    v0 = mesh.nodes[mesh.triangles[:, 0]]
    v1 = mesh.nodes[mesh.triangles[:, 1]]
    v2 = mesh.nodes[mesh.triangles[:, 2]]
    signed_area = 0.5 * ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    print(f"Aires signees : min = {signed_area.min():.6f}, max = {signed_area.max():.6f}")
    assert np.all(signed_area > 0), "Certains triangles ne sont pas orientes CCW !"
    print("Tous les triangles sont bien orientes dans le sens trigonometrique (CCW).")

    # --- Visuel : le maillage, les noeuds de bord, et la puce ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles, color="#1f77b4", linewidth=0.6)
    ax.scatter(mesh.nodes[boundary, 0], mesh.nodes[boundary, 1], color="black", s=12, zorder=5, label="noeuds de bord (Dirichlet)")

    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = p.CHIP_CENTER_X + p.CHIP_RADIUS * np.cos(theta)
    circle_y = p.CHIP_CENTER_Y + p.CHIP_RADIUS * np.sin(theta)
    ax.fill(circle_x, circle_y, color="#d62728", alpha=0.5, label="puce (source g)", zorder=6)

    ax.set_aspect("equal")
    ax.set_title(f"Maillage structure {mesh.nx}x{mesh.ny} ({mesh.n_elements} triangles)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    output_path = "figures/02_mesh_check.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"\nFigure enregistree : {output_path}")


if __name__ == "__main__":
    main()

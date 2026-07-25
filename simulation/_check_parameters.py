"""Script de verification du module 1 (parameters.py).

Ce script n'appartient pas au moteur final : c'est un test manuel pour
verifier, a l'oeil, que les valeurs de parameters.py sont sensees avant de
passer au module suivant (mesh.py).
"""

import numpy as np
import matplotlib.pyplot as plt

import parameters as p


def main() -> None:
    print("=== Domaine ===")
    print(f"Omega = [{p.DOMAIN.x_min}, {p.DOMAIN.x_max}] x [{p.DOMAIN.y_min}, {p.DOMAIN.y_max}]")
    print(f"Maillage cible : {p.MESH_NX} x {p.MESH_NY} cellules")

    print("\n=== Puce (source de chaleur) ===")
    print(f"Centre : ({p.CHIP_CENTER_X}, {p.CHIP_CENTER_Y})")
    print(f"Rayon (ecart-type gaussienne) : {p.CHIP_RADIUS}")
    print(f"Amplitude CHIP_POWER : {p.CHIP_POWER}")

    print("\n=== Viscosite ===")
    print(f"nu(T=0)  = {p.viscosity(0.0):.4f}")
    print(f"nu(T=1)  = {p.viscosity(1.0):.4f}")
    print(f"nu(T=3)  = {p.viscosity(3.0):.4f}")
    print(f"NU_REFERENCE = {p.NU_REFERENCE:.4f}")

    print("\n=== Nombre de Rayleigh : effet du bouton ===")
    for ra in [1e2, 1e3, 1e4, 1e5, 1e6]:
        coeff = p.buoyancy_coefficient(ra)
        print(f"Ra = {ra:>10.0e}  ->  rho0*beta*g = {coeff:.4f}")

    print(f"\nValeur active dans parameters.py : RAYLEIGH = {p.RAYLEIGH:.1e}")
    print(f"BUOYANCY_COEFFICIENT = {p.BUOYANCY_COEFFICIENT:.4f}")

    print("\n=== Temps ===")
    print(f"t_final = {p.TIME.t_final}, dt = {p.TIME.dt}, n_steps = {p.N_STEPS}")

    # --- Visuel 1 : la loi de viscosite nu(T) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    temperatures = np.linspace(0.0, 5.0, 200)
    axes[0].plot(temperatures, [p.viscosity(t) for t in temperatures], color="#1f77b4", linewidth=2)
    axes[0].axhline(0.1, color="grey", linestyle="--", linewidth=1, label="plancher nu_min = 0.1")
    axes[0].set_xlabel("Temperature T")
    axes[0].set_ylabel("Viscosite nu(T)")
    axes[0].set_title("Loi de viscosite : le fluide chaud s'ecoule plus facilement")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Visuel 2 : le domaine et la position de la puce ---
    rect_x = [p.DOMAIN.x_min, p.DOMAIN.x_max, p.DOMAIN.x_max, p.DOMAIN.x_min, p.DOMAIN.x_min]
    rect_y = [p.DOMAIN.y_min, p.DOMAIN.y_min, p.DOMAIN.y_max, p.DOMAIN.y_max, p.DOMAIN.y_min]
    axes[1].plot(rect_x, rect_y, color="black", linewidth=1.5, label="parois froides (u=0, T=0)")

    theta = np.linspace(0, 2 * np.pi, 100)
    for n_sigma, alpha_fill in zip([1, 2], [0.5, 0.2]):
        circle_x = p.CHIP_CENTER_X + n_sigma * p.CHIP_RADIUS * np.cos(theta)
        circle_y = p.CHIP_CENTER_Y + n_sigma * p.CHIP_RADIUS * np.sin(theta)
        axes[1].fill(circle_x, circle_y, color="#d62728", alpha=alpha_fill)
    axes[1].scatter([p.CHIP_CENTER_X], [p.CHIP_CENTER_Y], color="#d62728", zorder=5, label="puce (source g)")

    axes[1].set_xlim(p.DOMAIN.x_min - 0.05, p.DOMAIN.x_max + 0.05)
    axes[1].set_ylim(p.DOMAIN.y_min - 0.05, p.DOMAIN.y_max + 0.05)
    axes[1].set_aspect("equal")
    axes[1].set_title(f"Scenario : puce dans son boitier (Ra = {p.RAYLEIGH:.0e})")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    output_path = "figures/01_parameters_check.png"
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(output_path, dpi=130)
    print(f"\nFigure enregistree : {output_path}")


if __name__ == "__main__":
    main()

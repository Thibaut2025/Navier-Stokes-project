"""Parametres physiques et numeriques - convection naturelle autour d'une puce.

Scenario retenu : une puce electronique (source de chaleur interne) est
refroidie par convection naturelle a l'interieur d'un boitier aux parois
froides (modele de Boussinesq). Ce fichier centralise TOUTES les valeurs
numeriques du modele : les autres modules (mesh.py, matrices.py,
boundary_conditions.py, solver.py, ...) importent depuis ici et ne
contiennent aucune constante "en dur".

Correspondance avec le memoire (Doc.pdf p.1) :
    rho, alpha, nu(T), rho0*beta*g  -> parametres physiques de la forme forte
    Omega = [x_min,x_max] x [y_min,y_max] -> domaine de calcul
    g(x,t) -> source de chaleur (ici : la puce), construite dans
              boundary_conditions.py a partir des constantes CHIP_* ci-dessous
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# ---------------------------------------------------------------------------
# 1. Domaine de calcul (le boitier, vu en coupe 2D) et maillage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainParameters:
    """Rectangle de calcul Omega."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def length_x(self) -> float:
        return self.x_max - self.x_min

    @property
    def length_y(self) -> float:
        return self.y_max - self.y_min


# Domaine unite : garder L = 1 simplifie la formule du nombre de Rayleigh
# (voir buoyancy_coefficient ci-dessous). On pourra changer le rapport
# largeur/hauteur plus tard sans rien casser ailleurs.
DOMAIN = DomainParameters(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

# Nombre de subdivisions du maillage structure (chaque cellule -> 2 triangles).
# Compromis precision/vitesse : plus de mailles = solution plus fine mais
# systemes lineaires plus gros a resoudre a chaque pas de temps.
MESH_NX = 24
MESH_NY = 24


# ---------------------------------------------------------------------------
# 2. La puce : position et forme de la source de chaleur g(x, y)
# ---------------------------------------------------------------------------

# Centree en x, plaquee pres du bas du boitier (le composant est au fond).
CHIP_CENTER_X = 0.5 * (DOMAIN.x_min + DOMAIN.x_max)
CHIP_CENTER_Y = DOMAIN.y_min + 0.05 * DOMAIN.length_y

# Ecart-type de la gaussienne : la puce occupe une petite zone du fond.
CHIP_RADIUS = 0.06 * DOMAIN.length_x

# Amplitude de la source. Valeur de depart : sera recalibree au module 7
# (equation de la chaleur) une fois qu'on peut observer le T_max obtenu.
CHIP_POWER = 40.0


# ---------------------------------------------------------------------------
# 3. Nombre de Rayleigh : LE bouton de l'experience virtuelle
# ---------------------------------------------------------------------------
#
#   Ra = g * beta * DeltaT * L^3 / (nu * alpha)
#
#   Ra petit  (< ~1e3)   : la diffusion l'emporte, pas de mouvement organise
#   Ra moyen (1e3 - 1e5) : convection laminaire, boucles bien formees
#   Ra grand  (> ~1e6)   : regime instable, proche de la turbulence
#
# On travaille en variables sans dimension : DeltaT_ref = 1, beta = 1,
# alpha = 1, L = hauteur du domaine. Le nombre de Rayleigh cible fixe alors
# directement le produit rho0*beta*g qui pondere le terme de flottabilite
# M_flot (Doc.pdf / Navier_Stokes_et_Chaleur.pdf, eq. 14).

RAYLEIGH = 2.0e4

ALPHA = 1.0                      # diffusivite thermique de reference
RHO0 = 1.0                       # masse volumique de reference
BETA_THERMAL_EXPANSION = 1.0     # coefficient de dilatation thermique


def viscosity(temperature: float) -> float:
    """Viscosite cinematique dependante de la temperature : nu(T).

    Le fluide devient moins visqueux quand il chauffe (comme l'air pres
    d'une source chaude) : nu(T) = nu_min + (nu(0) - nu_min) * exp(-T),
    avec nu_min = 0.1 (plancher physique : la viscosite ne s'annule jamais)
    et nu(0) = 1.1.
    """

    return math.exp(-temperature) + 0.1


NU_REFERENCE = viscosity(0.0)  # nu(T=0), sert a calibrer Ra


def buoyancy_coefficient(rayleigh: float = RAYLEIGH) -> float:
    """Calcule rho0 * beta * g a partir du nombre de Rayleigh cible.

    En inversant Ra = g * beta * DeltaT * L^3 / (nu * alpha) avec
    DeltaT_ref = 1 et beta = 1 :

        g = Ra * nu(0) * alpha / L^3

    Le resultat rho0*beta*g est directement le coefficient utilise dans
    l'assemblage de M_flot (matrices.py).
    """

    length = DOMAIN.length_y
    g = rayleigh * NU_REFERENCE * ALPHA / (length ** 3)
    return RHO0 * BETA_THERMAL_EXPANSION * g


BUOYANCY_COEFFICIENT = buoyancy_coefficient(RAYLEIGH)


# ---------------------------------------------------------------------------
# 4. Critere de securite : la question a laquelle la simulation doit repondre
# ---------------------------------------------------------------------------

T_SAFE = 1.0  # seuil de temperature "critique" pour la puce (unites du modele)


# ---------------------------------------------------------------------------
# 5. Parametres temporels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeParameters:
    """Parametres du schema en temps (Euler implicite)."""

    t_final: float
    dt: float

    @property
    def n_steps(self) -> int:
        """Nombre de pas de temps deduit automatiquement de t_final et dt."""

        return int(round(self.t_final / self.dt))


TIME = TimeParameters(t_final=2.0, dt=0.01)
N_STEPS = TIME.n_steps

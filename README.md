# Navier-Stokes 2D, refroidissement d'un processeur

Projet en cours.

L'idée est de simuler l'air qui circule naturellement autour d'une puce électronique, pour vérifier qu'elle reste sous sa température critique sans ventilation forcée.

Le modèle couple les équations de Navier-Stokes incompressibles à l'équation de la chaleur. Le couplage passe par l'approximation de Boussinesq : la densité de l'air n'est traitée comme variable que dans le terme de flottabilité, ce qui suffit à faire monter l'air chaud sans avoir à résoudre le cas compressible.

Côté discrétisation, j'utilise des éléments finis de Taylor-Hood, du P2 pour la vitesse et la température et du P1 pour la pression. Ce choix n'est pas cosmétique : un couple d'ordre égal ne vérifie pas la condition inf-sup et fait apparaître des oscillations parasites sur la pression. Le temps est traité en différences finies.

Python, NumPy.

# Simulation Navier-Stokes + Chaleur — Rapport de projet

État au 25 juillet 2026 — étapes 1 à 4 terminées sur 10.

---

## 1. Ce qu'on simule

### Le scénario physique

Une **puce électronique** dégage de la chaleur à l'intérieur d'un **boîtier fermé**
dont les parois sont froides. L'air (ou le fluide) autour de la puce chauffe, se
dilate, devient plus léger, et **monte**. En haut, il touche les parois froides,
se refroidit, redescend. Ce mouvement circulaire qui s'auto-entretient s'appelle
la **convection naturelle** : personne ne souffle sur le système, c'est la
chaleur elle-même qui met le fluide en mouvement.

La question à laquelle la simulation doit répondre : **la puce reste-t-elle sous
sa température critique**, ou faut-il un refroidissement actif ?

### Le modèle mathématique

On résout deux équations **couplées** sur un domaine carré 2D `Ω = [0,1]×[0,1]`
(le boîtier vu en coupe) :

1. **Navier-Stokes incompressible** (mouvement du fluide) — avec l'approximation
   de **Boussinesq** : le fluide est traité comme incompressible partout, sauf
   dans un unique terme de flottabilité `ρ₀·β·g·T` qui pousse le fluide chaud
   vers le haut. C'est l'astuce standard pour traiter la convection naturelle
   sans avoir à résoudre un écoulement compressible.
2. **Équation de la chaleur** (transport de la température) — diffusion +
   convection par le fluide + source de chaleur `g(x,y)` (la puce, modélisée par
   une gaussienne près du bas du domaine).

Le couplage est **bidirectionnel** : la température fait bouger le fluide (par la
flottabilité), et le fluide déplace la chaleur (par la convection). C'est ce qui
rend le problème intéressant — et non linéaire.

### Les trois champs inconnus

| Champ | Signification | Discrétisation |
|---|---|---|
| **u** = (u_x, u_y) | vitesse du fluide en chaque point | P2 (quadratique) |
| **p** | pression | P1 (linéaire) |
| **T** | température | P2 (quadratique) |

### Le nombre de Rayleigh : le bouton de l'expérience

Tout le comportement du système est piloté par un seul nombre sans dimension :

```
Ra = g·β·ΔT·L³ / (ν·α)
```

- `Ra < 10³` — la diffusion l'emporte, la chaleur s'évacue sans mouvement organisé
- `Ra ≈ 10³ – 10⁵` — **convection laminaire**, de belles boucles de recirculation
- `Ra > 10⁶` — régime instable, proche de la turbulence

Valeur retenue : **Ra = 2·10⁴** (convection laminaire bien formée). Changer cette
seule constante dans `parameters.py` change tout le régime de l'écoulement.

Particularité du modèle : la **viscosité dépend de la température**,
`ν(T) = exp(−T) + 0.1`. Le fluide chaud est moins visqueux, donc s'écoule plus
facilement — avec un plancher à 0.1 pour que la viscosité ne s'annule jamais.

### Pourquoi l'élément de Taylor-Hood (P2/P1)

On ne peut pas choisir n'importe quelle combinaison vitesse/pression. Si vitesse
et pression sont discrétisées au même ordre, le système devient instable et fait
apparaître des **modes de pression parasites** (des oscillations en damier qui
n'ont aucun sens physique).

L'élément de **Taylor-Hood** — vitesse un degré plus riche que la pression
(P2 pour u, P1 pour p) — satisfait la condition mathématique dite **inf-sup**
(ou condition LBB) qui garantit la stabilité. C'est le choix classique et sûr.

Conséquence pratique : le maillage porte **deux familles de nœuds**. Les sommets
des triangles (pour la pression), et les sommets **plus les milieux d'arêtes**
(pour la vitesse et la température).

Sur le maillage actuel (24×24 cellules → 1152 triangles) :

| | Nombre de degrés de liberté |
|---|---|
| Pression (P1, sommets) | 625 |
| Température, ou une composante de vitesse (P2) | 2 401 |
| Vitesse complète (2 composantes P2) | 4 802 |

---

## 2. État d'avancement

| Étape | Description | État | Fichier |
|---|---|---|---|
| 1 | Paramètres physiques et numériques | ✅ fait | `parameters.py` |
| 2 | Construction du maillage | ✅ fait | `mesh.py` |
| 3 | Éléments finis (formes, jacobien, matrices locales) | ✅ fait | `finite_elements.py` |
| 4 | **Assemblage global des matrices** | ✅ **fait** | `matrices.py` |
| 5 | Conditions aux limites (Dirichlet u=0, T=0) | ⬜ à faire | — |
| 6 | Système Navier-Stokes sous forme matricielle | ⬜ à faire | — |
| 7 | Équation de la chaleur (diffusion + convection + source) | ⬜ à faire | — |
| 8 | Couplage NS ↔ chaleur | ⬜ à faire | — |
| 9 | Boucle en temps (Euler implicite) | ⬜ à faire | — |
| 10 | Post-traitement (lignes de courant, cartes de T, animation) | ⬜ à faire | — |

---

## 3. Description détaillée de chaque fichier

Tout le code de la simulation vit dans le dossier `simulation/`. Les fichiers
préfixés par `_check_` ne font **pas** partie du moteur : ce sont des scripts de
vérification manuelle, exécutés à la fin de chaque étape pour valider le travail
avant de passer à la suivante.

---

### `parameters.py` — toutes les constantes du modèle

Le fichier central : **aucun autre module ne contient de valeur numérique en
dur**. Tout vient d'ici, donc changer le scénario ne demande de toucher qu'un
seul fichier.

Contenu :

- **`DomainParameters`** (dataclass gelée) et l'instance **`DOMAIN`** — le
  rectangle de calcul `[0,1]×[0,1]`, avec les propriétés `length_x` / `length_y`.
- **`MESH_NX`, `MESH_NY`** = 24 — finesse du maillage.
- **`CHIP_CENTER_X/Y`, `CHIP_RADIUS`, `CHIP_POWER`** — position, taille et
  puissance de la puce. Elle est centrée horizontalement et plaquée près du bas
  du boîtier (à 5 % de la hauteur), avec un rayon de 6 % de la largeur.
- **`RAYLEIGH`** = 2·10⁴, plus `ALPHA`, `RHO0`, `BETA_THERMAL_EXPANSION` — les
  paramètres physiques adimensionnés.
- **`viscosity(T)`** — la loi `ν(T) = exp(−T) + 0.1` décrite plus haut, et
  `NU_REFERENCE = ν(0)` qui sert à calibrer le Rayleigh.
- **`buoyancy_coefficient(Ra)`** — inverse la formule du Rayleigh pour en déduire
  le coefficient `ρ₀·β·g` qui pondérera le terme de flottabilité. C'est le pont
  entre « je veux tel régime d'écoulement » et « quelle constante mettre dans la
  matrice ».
- **`T_SAFE`** = 1.0 — le seuil de température critique de la puce : le critère
  de sécurité auquel la simulation devra répondre.
- **`TimeParameters`** et **`TIME`** — `t_final = 2.0`, `dt = 0.01`, d'où
  `N_STEPS = 200` pas de temps (déduit automatiquement, pas saisi à la main).

---

### `mesh.py` — le squelette géométrique

Transforme le rectangle continu en un ensemble fini de nœuds reliés par des
triangles. Stratégie : **maillage structuré**. On découpe le domaine en
`nx × ny` cellules rectangulaires identiques, puis chaque cellule est coupée en
deux triangles par sa diagonale. C'est le maillage le plus simple possible —
largement suffisant pour un domaine rectangulaire.

Contenu :

- **`Mesh2D`** (dataclass gelée) — le maillage complet :
  - `nodes` : tableau `(n_nodes, 2)` des coordonnées `(x, y)`
  - `triangles` : tableau `(n_elements, 3)` des indices des 3 sommets de chaque
    triangle, **rangés dans le sens trigonométrique (CCW)** — une convention
    respectée partout ensuite, car elle garantit un déterminant jacobien positif
  - `nx, ny, dx, dy` : la grille et ses pas
  - `boundary_nodes` : dictionnaire `{"left", "right", "bottom", "top"}` → indices
    des nœuds de chaque bord, pour les conditions de Dirichlet de l'étape 5
  - propriétés `n_nodes` et `n_elements`
- **`node_index(i, j, nx)`** — convertit un indice de grille `(i, j)` en indice
  global de nœud (même logique qu'un `numpy.ravel` en ordre C).
- **`build_structured_triangular_mesh(nx, ny)`** — la fonction principale :
  construit les nœuds, la connectivité des triangles, et les listes de bord.
- **`all_boundary_dofs(mesh)`** — renvoie tous les nœuds de bord sans doublons
  (les 4 coins appartiennent à deux côtés chacun, il faut dédupliquer).

---

### `finite_elements.py` — le cœur mathématique de la méthode

Le lien entre la géométrie discrète et les intégrales à calculer. Quatre briques.

**1. Le triangle de référence et les fonctions de forme**

Tous les calculs se font sur un unique triangle de référence
`K̂ = {(ξ,η) : ξ≥0, η≥0, ξ+η≤1}`, puis sont transportés vers chaque triangle réel.

- `REFERENCE_NODES_P1` / `REFERENCE_NODES_P2` — les 3 sommets, et les 3 sommets
  plus les 3 milieux d'arêtes (ordre `v0, v1, v2, m01, m12, m20`)
- **`reference_p1_basis(ξ, η)`** — les 3 fonctions de forme linéaires, qui sont
  exactement les coordonnées barycentriques `L1, L2, L3`
- **`reference_p1_gradients()`** — leurs gradients, constants
- **`reference_p2_basis(ξ, η)`** — les 6 fonctions quadratiques :
  `Lₖ(2Lₖ−1)` sur les sommets, `4LᵢLⱼ` sur les milieux d'arêtes
- **`reference_p2_gradients(ξ, η)`** — leurs gradients, qui eux varient dans le
  triangle

**2. Le changement de variables vers le triangle physique**

- **`TriangleGeometry`** (dataclass gelée) — pour un triangle réel : ses sommets,
  le jacobien `J` de la transformation (constant, car la transformation est
  affine), son déterminant, son inverse, et l'aire
- **`triangle_geometry(vertices)`** — calcule tout ça, et lève une erreur sur un
  triangle dégénéré
- **`physical_gradients(ref_grads, inv_J)`** — applique la règle de la chaîne
  `∇ₓφ = J⁻ᵀ ∇̂φ` pour transporter les gradients

**3. La quadrature numérique**

- **`reference_triangle_quadrature()`** — règle de **Dunavant à 7 points, exacte
  pour les polynômes de degré ≤ 5**. Nécessaire dès qu'on manipule du P2 : les
  produits de fonctions quadratiques ne s'intègrent plus par une formule fermée
  commode. Les poids sont déjà mis à l'échelle de l'aire du triangle de référence
  (leur somme vaut 1/2).

**4. Les matrices élémentaires**

- **`local_mass_matrix_p1`** / **`local_stiffness_matrix_p1`** — formules fermées
  classiques (les intégrandes P1 sont de bas degré, la quadrature est inutile)
- **`local_mass_matrix_p2`** / **`local_stiffness_matrix_p2`** — calculées par
  quadrature

**5. L'enrichissement P1 → P2**

- **`QuadraticMesh2D`** (dataclass gelée) — le maillage enrichi : les nœuds P1
  d'origine **suivis** des milieux d'arêtes ajoutés, la connectivité `(n_elem, 6)`,
  et le dictionnaire `edge_to_midpoint`
- **`build_quadratic_mesh(mesh)`** — ajoute un nœud au milieu de chaque arête.
  Le dictionnaire garantit qu'une arête partagée par deux triangles ne reçoit
  **qu'un seul** nœud milieu, correctement partagé — c'est ce qui assure la
  continuité de la vitesse entre éléments voisins.

---

### `matrices.py` — assemblage global (étape 4, nouveau)

Le module précédent sait calculer les petites matrices sur **un** triangle. Ici
on parcourt **tous** les triangles et on dépose chaque contribution locale à la
bonne place dans une grande matrice globale creuse.

C'est le principe mécanique de la méthode : une intégrale sur `Ω` est la somme
des intégrales sur chaque triangle, donc la matrice globale est la somme des
matrices locales « dispatchées » par la table de connectivité.

**Convention de rangement de la vitesse** (importante pour la suite) : le vecteur
global de vitesse est rangé **par composante**,
`v = [vx_0 … vx_{n−1} | vy_0 … vy_{n−1}]`, de taille `2 × n_p2`. Ce rangement par
blocs rend le système de Stokes lisible : les blocs de diffusion sont alors deux
copies de `K_p2` sur la diagonale.

Contenu :

- **`_scatter(...)`** et **`_to_csr(...)`** — les deux utilitaires d'assemblage.
  On accumule les contributions sous forme de triplets `(ligne, colonne, valeur)`,
  puis on convertit en CSR. Le passage par le format COO est volontaire : c'est
  lui qui **additionne automatiquement les doublons**, c'est-à-dire les
  contributions de plusieurs triangles à un même couple de nœuds.
- **`assemble_mass_p1`, `assemble_stiffness_p1`** — masse et rigidité sur les
  sommets (espace de la pression), `625 × 625`
- **`assemble_mass_p2`, `assemble_stiffness_p2`** — masse et rigidité sur les
  nœuds P2 (vitesse et température), `2401 × 2401`
- **`local_divergence_matrix(vertices)`** — matrice locale `(3, 6, 2)` :
  `∫ φ1ₖ · ∂φ2ⱼ/∂x_d`. Intégrande de degré 2, la quadrature d'ordre 5 est exacte.
- **`assemble_divergence(mesh, quadratic)`** — la matrice **B** de couplage
  vitesse-pression, `625 × 4802`. Elle discrétise le terme `∫ q·div(v)` de la
  formulation faible. Sa transposée `−Bᵀ` fournira le terme de gradient de
  pression dans l'équation de quantité de mouvement : c'est la **structure en
  point-selle** caractéristique de Taylor-Hood.
- **`local_convection_matrix(vertices, element_velocity)`** — matrice locale
  `(6,6)` du transport. **Note sur la quadrature** : l'intégrande est de degré
  2 (u en P2) + 1 (gradient de P2) + 2 (fonction test P2) = **5**, donc la règle
  de Dunavant d'ordre 5 déjà disponible l'intègre **exactement** — rien à ajouter.
- **`assemble_convection(quadratic, velocity)`** — la matrice **N(u)**,
  `2401 × 2401`. C'est la **seule matrice qui dépend de la solution** : elle
  devra être réassemblée à chaque pas de temps (linéarisation semi-implicite,
  on prend `u` au pas précédent). Elle est **non symétrique** — le transport a un
  sens. Le même opérateur sert pour le terme non linéaire `(u·∇)u` de
  Navier-Stokes et pour le transport `u·∇T` de l'équation de la chaleur.
- **`GlobalMatrices`** (dataclass gelée) — regroupe tout ce qui est assemblé
  **une fois pour toutes** (les deux maillages + les 5 matrices constantes), avec
  les propriétés `n_pressure_dofs`, `n_scalar_dofs`, `n_velocity_dofs`.
  `N(u)` en est volontairement exclue, puisqu'elle change à chaque pas.
- **`assemble_all(mesh=None)`** — construit tout d'un coup ; sans argument, prend
  le maillage par défaut de `parameters.py`.

---

### Les scripts de vérification

#### `_check_parameters.py`
Affiche toutes les valeurs de `parameters.py` (domaine, puce, viscosité à
plusieurs températures, effet du Rayleigh sur le coefficient de flottabilité,
paramètres temporels) et produit une figure à deux panneaux : la **loi de
viscosité ν(T)**, et une vue du **domaine avec la position de la puce**.

#### `_check_mesh.py`
Vérifie les dimensions du maillage, le compte de nœuds de bord (qui doit égaler
le périmètre en nombre de segments), et surtout que **tous les triangles sont
orientés dans le sens trigonométrique** (aires signées toutes positives). Produit
une figure du maillage avec les nœuds de Dirichlet et la puce.

#### `_check_matrices.py`
Vérifie les matrices assemblées par des **identités mathématiques que toute
matrice élément fini correcte doit satisfaire** — ce qui en fait un filet de
sécurité indépendant de toute solution de référence :

| Test | Ce qu'il attrape |
|---|---|
| Nombre de nœuds P2 = sommets + arêtes (relation d'Euler) | erreur dans l'enrichissement du maillage |
| Symétrie de masse et rigidité | erreur d'indices dans l'assemblage |
| Somme des coefficients de masse = aire du domaine | erreur de jacobien ou de poids de quadrature |
| `K @ 1 = 0` (un champ constant a une énergie nulle) | erreur de signe ou de transport des gradients |
| `B @ v_uniforme = 0` (un champ uniforme est incompressible) | erreur dans la matrice de divergence |
| `N(u) @ 1 = 0` pour toute vitesse | erreur dans la convection |
| `N(u)` **non** symétrique | terme de transport perdu ou symétrisé par erreur |

Résultats obtenus : toutes les erreurs sont au niveau de la précision machine
(10⁻¹⁵ à 10⁻¹⁷). Produit une figure des **motifs de remplissage** (sparsity
patterns) des 4 matrices principales.

---

## 4. Ce qui est codé à la main vs les modules Python utilisés

### Aucune bibliothèque d'éléments finis n'est utilisée

**Tout l'algorithme éléments finis est écrit depuis zéro.** Le projet n'utilise
ni FEniCS, ni scikit-fem, ni FreeFEM, ni aucun autre solveur ou framework EF.
Les bibliothèques importées ne servent qu'à des tâches génériques : algèbre
linéaire, stockage creux, tracé.

### Écrit entièrement à la main

| Composant | Où | Ce que ça représente |
|---|---|---|
| Génération du maillage triangulaire structuré | `mesh.py` | nœuds, connectivité CCW, détection des bords |
| Fonctions de forme P1 (3) et P2 (6) | `finite_elements.py` | les polynômes de base, écrits explicitement |
| Gradients des fonctions de forme | `finite_elements.py` | dérivées calculées à la main, en dur |
| Transformation vers le triangle de référence | `finite_elements.py` | jacobien, déterminant, règle de la chaîne `J⁻ᵀ∇̂φ` |
| Règle de quadrature de Dunavant (7 points, ordre 5) | `finite_elements.py` | points et poids tabulés explicitement |
| Matrices élémentaires (masse, rigidité, divergence, convection) | `finite_elements.py`, `matrices.py` | les intégrales locales |
| Enrichissement P1 → P2 (nœuds milieux d'arêtes partagés) | `finite_elements.py` | la construction de l'espace de Taylor-Hood |
| Boucle d'assemblage global (scatter des triplets) | `matrices.py` | le dispatch local → global |
| Calibration physique (Rayleigh → coefficient de flottabilité) | `parameters.py` | l'inversion de la formule du Rayleigh |

### Modules Python utilisés, et à quoi exactement

| Module | Usage précis dans ce projet |
|---|---|
| **`numpy`** | Algèbre linéaire **dense sur les petites matrices locales** (3×3, 6×6, 6×2), géométrie des triangles (`np.linalg.det` et `np.linalg.inv` sur le jacobien 2×2), et tous les tableaux de nœuds et de connectivité. Aussi `np.einsum` et `np.outer` pour écrire les intégrandes locales de façon compacte. |
| **`scipy.sparse`** | **Stockage et assemblage des matrices globales creuses uniquement.** Le format `coo_matrix` sert à accumuler les triplets et à sommer automatiquement les doublons, `tocsr()` donne le format final efficace pour les produits matrice-vecteur. Aucun solveur SciPy n'est encore utilisé (ce sera l'étape 9, avec `scipy.sparse.linalg`). |
| **`matplotlib`** | **Uniquement les figures de vérification** — aucun rôle dans le calcul. `triplot` pour le maillage, `spy` pour les motifs de remplissage, tracés classiques pour la loi de viscosité. |
| **`dataclasses`** | Structures de données immuables (`frozen=True`) pour regrouper proprement les objets du modèle : `DomainParameters`, `TimeParameters`, `Mesh2D`, `TriangleGeometry`, `QuadraticMesh2D`, `GlobalMatrices`. Zéro logique métier, juste du regroupement lisible. |
| **`math`** | Une seule fonction, `math.exp`, dans la loi de viscosité `ν(T)`. |
| **`os`** | Création du dossier `figures/` dans les scripts de vérification. |
| **`__future__.annotations`** | Confort d'écriture des annotations de type (`Mesh2D \| None`, etc.). |

---

## 5. Comment lancer les vérifications

Les scripts s'exécutent **depuis le dossier `simulation/`** (ils importent leurs
modules voisins directement et écrivent dans `figures/`) :

```bash
cd simulation
python _check_parameters.py
python _check_mesh.py
python _check_matrices.py
```

Chacun affiche un rapport chiffré dans le terminal, vérifie ses hypothèses par
des `assert` (le script s'arrête net si quelque chose cloche), et enregistre une
figure.

### Ce que montre chaque figure

| Figure | Contenu |
|---|---|
| `figures/01_parameters_check.png` | **Gauche** : la loi de viscosité `ν(T)`, décroissante, avec son plancher à 0.1. **Droite** : le domaine carré, les parois froides, et la puce (zones à 1σ et 2σ de la gaussienne) plaquée en bas au centre. |
| `figures/02_mesh_check.png` | Le maillage triangulaire structuré 24×24 (1152 triangles), les nœuds de bord marqués en noir (là où seront imposées les conditions de Dirichlet `u=0, T=0`), et la puce en rouge. |
| `figures/03_matrices_check.png` | Les **motifs de remplissage** des 4 matrices globales (masse P1, rigidité P2, divergence B, convection N(u)). La structure en bandes reflète la numérotation des nœuds ; le très faible remplissage (0.5 à 1 %) montre l'intérêt du stockage creux. |

### Dépendances

```
python >= 3.10   (la syntaxe `zip(..., strict=True)` et `X | None` est utilisée)
numpy
scipy
matplotlib
```

Versions testées : Python 3.11.9, numpy 2.1.3, scipy 1.15.3, matplotlib 3.10.3.

---

## 6. Prochaine étape

**Étape 5 — conditions aux limites.** Imposer `u = 0` (adhérence) et `T = 0`
(parois froides) sur tout le bord du domaine, c'est-à-dire éliminer ou contraindre
les degrés de liberté correspondants dans les matrices assemblées ici. Les nœuds
concernés sont déjà identifiés par `mesh.boundary_nodes` et `all_boundary_dofs`
pour les sommets P1 ; il faudra étendre cette détection aux **milieux d'arêtes de
bord** pour les degrés de liberté P2.

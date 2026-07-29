# Simulation Navier-Stokes 2D — Refroidissement de processeurs

> ⚠️ **Projet en cours.** Le code et les résultats évoluent encore.

## Objectif

Simuler la convection naturelle de l'air autour d'une puce électronique afin de vérifier
que celle-ci reste sous sa température critique de fonctionnement.

## Modèle

- Équations de **Navier-Stokes incompressibles** couplées à l'**équation de la chaleur**
- Couplage thermique par l'**approximation de Boussinesq** (la variation de densité
  n'intervient que dans le terme de flottabilité)
- Domaine 2D représentant la puce et l'air environnant

## Discrétisation

- **Éléments finis de Taylor-Hood** : P2 pour la vitesse et la température, P1 pour la pression
  (condition inf-sup satisfaite)
- **Différences finies** en temps

## Stack

`Python` · `NumPy` · éléments finis · EDP

---

Auteur : **Thibaut TCHINHOUN** — élève ingénieur en Génie Mathématique et Modélisation (ENSGMM, UNSTIM Abomey)
Portfolio : https://tchinhounthibautportfolio.vercel.app/

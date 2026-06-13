# Article conventions

## Class & preamble
- `\documentclass[11pt]{article}` + `geometry` (a4, 1in margins).
- Author block via `authblk`: `\author[n]{First Last}` + `\affil[n]{...}`.
- Math: `amsmath, amssymb, physics` (\ket,\bra,\expval,\abs) and/or `braket`.
- `booktabs` for tables, `hyperref` (colorlinks) for links, `lmodern` font.

## Structure (this paper)
1. Introduction (eigenstate-prep bottleneck; VQE/adiabatic/QPE+rodeo limits; the
   hybrid idea; pairing model as testbed).
2. The pairing model and its qubit encoding.
3. The hybrid pipeline (coarse UCCSD-VQE; prolongation; resolution refinement;
   rodeo algorithm with Eqs. 1-2 and the convergence criterion).
4. Results (UCCSD vs FCI/CCD + scaling; refinement overlap & energy; rodeo scan
   + convergence).
5. Discussion & outlook (hardware/Trotter/Magne; applications).

## Authors (alphabetical by surname) — keep in sync with the application
Bogner (FRIB/MSU), Glittum (Oslo), Hergert (FRIB/MSU), Hjorth-Jensen (Oslo),
Lange (Oslo), LaRose (MSU), Lee (FRIB/MSU), Massel (USN/Oslo).

## Figure captions
Self-contained, describe axes and the takeaway. Reference exact numbers
(e.g. peak at E=0.6352 vs exact 0.635548) only if a notebook cell printed them.

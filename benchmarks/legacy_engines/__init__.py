"""The retired engines — benchmark OPPONENTS, not the library.

These are the pre-redesign solvers, moved out of latentspace.universal when
Daniel's specified GA (latentspace/universal/ga.py) became the one API:

    explorer  — per-individual decoder evolution (holds the single-fitness
                records: apple 0.00178, the TSP wins)
    solver    — its explore -> distill -> exploit orchestration (solve_single)
    distill   — PCA compression of vetted solutions into a latent space
    exploit   — the latent-space GA exploit phase
    multi     — the champion-per-problem shared-decoder population (solve_many)
    cma       — self-contained CMA-ES, always a baseline, never a component

They exist so every historical benchmark reproduces and so the new design
always has its record-holding opponents on the bench. Nothing here is API.
"""

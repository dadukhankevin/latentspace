"""Finch 4 — the name this library is becoming.

`latentspace` was the working title of the research campaign; Finch 4
is the unification of that work with the Finch lineage
(github.com/dadukhankevin/Finch). This package is the forward-facing
import: everything the library does, one namespace.

    import finch4

    finch4.solve(...)                  # the vetted tensor engine
    finch4.AgenticGA(...)              # the agentic substrate's engine
    finch4.Environment([...layers])    # composition, Finch-style
    finch4.classic                     # traditional GA layers
    finch4.live_progress()             # every run on the dashboard

The implementation still lives under `latentspace.*` while the rename
completes; this namespace is the stable way to write new code."""

from latentspace.finch import (AskRun, Audit, Consolidate, Environment,
                               Layer, SolveWhole, agentic_environment,
                               tensor_environment)
from latentspace.finch import classic
from latentspace.universal import (AgenticGA, Distillation, GAResult,
                                   ProblemResult, live_progress,
                                   register_architecture,
                                   register_substrate, solve)

__all__ = [
    "solve", "GAResult", "ProblemResult", "AgenticGA", "Distillation",
    "register_architecture", "register_substrate", "live_progress",
    "Environment", "Layer", "AskRun", "Audit", "Consolidate",
    "SolveWhole", "agentic_environment", "tensor_environment", "classic",
]

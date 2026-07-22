"""CLIP ecosystem: every prompt is a SPECIES, all alive at once — Daniel's v3.

v2's seats/rotation/hibernation existed only because fewer prompts than
niches could run at a time. That premise was false: a generation's cost
scales with the number of CHILDREN, not species — batch every child through
ONE CLIP vision pass and score each against its own species' text vector.
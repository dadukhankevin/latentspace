the base methodology, BUT replace local greedy match decisions with a finite-horizon cost model: at every position, compare a literal with every available back-reference and choose the path whose estimated arithmetic-coded token cost plus recursively optimal suffix cost is smallest, with distance, length, flag, and literal costs included rather than maximizing match length alone; use adaptive finite-context coding for literals and retain a self-contained fallback when matches do not pay for themselves.

contradicts_base: false

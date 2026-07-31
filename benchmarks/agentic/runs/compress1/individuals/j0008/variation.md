The base methodology, BUT replace any fixed model rescaling threshold with a concentration-adaptive threshold computed from each context's total mass and dominant-symbol share, capped only by the 16-bit safety limit. Concentrated contexts therefore forget stale counts sooner while diverse contexts retain more history; the adaptive rescaling threshold is recomputed during encoding and decoding so the byte-stream interface remains exactly lossless.

contradicts_base: false

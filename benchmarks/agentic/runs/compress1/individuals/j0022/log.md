# j0022 compression log

The slice is structured text with many repeated phrases and strong order-2/order-3 byte contexts. The artifact exploits both: a field-aware dynamic-programming LZ parse handles sufficiently long repeats, while PPM backoff with context exclusion models literal bytes and also updates across copied bytes.

## Canonical iterations

1. Initial range-coded LZ/PPM artifact, without context exclusion, literal planning scale 0.60. Canonical output:

   `{"task": "compress", "score": -4.377685546875, "bpb": 4.37769, "seconds": 0.91, "holdout": false, "errors": []}`

2. Added context exclusion to PPM backoff and calibrated planning scale to 0.12. Canonical output:

   `{"task": "compress", "score": -3.166015625, "bpb": 3.16602, "seconds": 1.23, "holdout": false, "errors": []}`

3. Calibrated the PPM escape frequency to 3 and planning scale to 0.21. Canonical output:

   `{"task": "compress", "score": -3.0400390625, "bpb": 3.04004, "seconds": 0.9, "holdout": false, "errors": []}`

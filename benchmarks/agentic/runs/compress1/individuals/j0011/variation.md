the base methodology, BUT retain the global Burrows–Wheeler and move-to-front transform while replacing the single flat rank histogram with typed events: adaptively code run/nonrun flags conditioned on the prior event type, code nonzero ranks with an exact previous-rank context plus adaptive unigram backoff, and code zero-run lengths in a separate adaptive histogram; synchronize these integer models in both directions and store only the original length, BWT primary index, and arithmetic payload.

contradicts_base: false

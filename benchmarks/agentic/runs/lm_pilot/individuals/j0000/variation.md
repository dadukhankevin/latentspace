The base methodology, BUT step 2's "structure to exploit" must come
from measuring the exact distribution the scorer's instance generator
draws from — the fixed data.txt corpus and its split arithmetic — and
the artifact must be specialized to those measurements rather than to
byte-modeling in general: model support restricted to the alphabet
actually observed in the train split (119 of 256 bytes), the output
head initialized at the train unigram distribution, and the corpus's
verbatim self-repetition treated as first-class structure via
exact-match retrieval (rolling-hash n-gram indexes over the train
corpus, plus a causal online index over the eval prefix) blended into
the neural distribution by match length and count mass.

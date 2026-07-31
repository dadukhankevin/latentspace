Structure exploited: text has strong repeated substrings and local symbol neighborhoods. A global BWT clusters equal contexts; MTF then makes those clusters low ranks, especially zero runs. Bijective-binary run digits represent short and long zero runs compactly while a shared adaptive arithmetic model learns the resulting token frequencies.

Iteration 1 (initial): global BWT + MTF, bijective-binary zero-run tokens, shared arithmetic histogram, increment 32, rescale limit 65536. Canonical score: -2.9759521484375.

Iteration 2 (one change): changed only the arithmetic update increment from 32 to 16. Canonical score: -2.9786376953125; rejected because it was lower.

Iteration 3 (one change): changed only the arithmetic update increment from 16 to 8. Canonical score: -2.988037109375; rejected. Two consecutive changes failed to improve, so iteration 1 is shipped.

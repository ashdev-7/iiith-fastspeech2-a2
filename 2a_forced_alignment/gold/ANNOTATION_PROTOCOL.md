# Hand-labelled reference: annotation protocol

**Why.** No aligner output can be checked without a reference. Hand labels by a trained person are the
standard reference in the field: TIMIT and Buckeye are labelled this way. Trained phoneticians agree on
about 93 % of phone boundaries within 20 ms (Hosom 2000, as cited by Yuan et al., Interspeech 2013), and
MFA 3.0's mean phone-boundary error on hand-labelled TIMIT is about 12 ms (McAuliffe et al., 2026, arXiv
2606.18466). A non-specialist annotator will be less precise than that. So this reference is designed
to resolve errors of tens to hundreds of milliseconds (the pause errors), and it measures its own
precision instead of assuming it.
This protocol measures *our own* error too: every word is marked twice, blind, and the two passes are
compared. Aligner differences smaller than that pass-to-pass difference are reported as unresolvable.

**What.** 27 words = 54 boundaries (`kit/words_to_mark.txt`):
* all 15 words of the 3 ISLE prompts;
* 12 words drawn **at random** (fixed seed) from the news speech, so nobody picked easy cases.

The kit shows only each word and the whole second after which it starts. It never shows an aligner
boundary.

## Rules (decide before you start, apply them identically every time)
1. **Blind.** Do not open anything in `outputs/` or `analysis/figures/` until both passes are finished.
2. **Two passes, at least one day apart.** Pass 2 is done from the empty TextGrids in `kit/pass2/`, without
   looking at pass 1.
3. **Use the waveform *and* the spectrogram.** Zoom until about 0.3–0.5 s fills the window. Listen to
   the selection (Tab) as often as you want.
4. **Word start** = the first acoustic evidence of its first sound:
   * vowel or sonorant: where its formants (dark horizontal bands) appear;
   * fricative (s, f, sh): where the high-frequency noise begins;
   * stop (p, t, k, b, d, g) after silence: the release burst (a thin vertical spike);
   * stop after speech: the start of the closure (where the previous sound's energy stops).
5. **Word end** = the last acoustic evidence of its last sound: the end of the formants or frication.
   For a final stop, include the release burst if there is one.
6. **Between two words with no pause**, the end of one word is the start of the next: put one boundary.
7. **Put boundaries on a zero crossing** of the waveform where possible (Praat: Ctrl+0 moves the cursor
   to the nearest zero crossing).

## How to do it in Praat (about 45–60 min per pass)
1. Open Praat. *Open → Read from file…* → `corpus/<speaker>/<file>.wav`. Then *Open → Read from file…* →
   `gold/kit/pass1/<file>.TextGrid` (an empty tier called `gold`).
2. Select both objects in the object list (Ctrl+click), then click **View & Edit**.
3. Go to the second given in `words_to_mark.txt` and find the word by listening.
4. Click at the word start on the **gold** tier (the top-right circle on the tier adds a boundary there),
   or click in the waveform and press **Enter**. Do the same at the word end.
5. Click inside the new interval and type the word exactly as listed (lower case).
6. Repeat for every listed word of the file, **in the listed order**. Leave everything else empty.
7. Save: *File → Save TextGrid as text file…* and **overwrite** `gold/kit/pass1/<file>.TextGrid`.
8. Do the same for all 6 files. A day later, repeat with `gold/kit/pass2/`.

## Then
```
python scripts/score_gold.py      # -> analysis/gold_check.md, analysis/gold_errors.csv
```
The script checks that every listed word was labelled, in order. It reports:
* your own consistency (pass 1 vs. pass 2);
* each experiment's error against the mean of your two passes;
* a paired comparison against `arpa_base`.

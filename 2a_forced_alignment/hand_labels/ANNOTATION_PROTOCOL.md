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

**What.** 60 words = 120 boundaries (`kit/words_to_mark.txt`):
* all 15 words of the 3 ISLE prompts;
* 45 words drawn **at random** (fixed seed) from the news speech, so nobody picked easy cases.
  (The first kit had 12; it was enlarged to 45 before any word was marked, to narrow the intervals.)

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

## Easiest way: the browser tool (recommended)
`python scripts/marking_tool.py build` writes `hand_labels/kit/marking_tool.html` (the six recordings are
embedded; it shows no aligner output). Open it in a browser. Each word gets one screen with the
waveform and a spectrogram: click the start, click the end, play to check, zoom to fine-tune, **Next**.
Progress is kept in the browser. At the end it downloads `passN_marks.json`; then
`python scripts/marking_tool.py convert passN_marks.json` writes `kit/passN/*.TextGrid`
(boundaries moved to a zero crossing only if one lies within 2.5 ms), the same files the Praat routes give.

## Guided Praat script
In Praat: **Open → Open Praat script…** → `hand_labels/kit/mark.praat` → **Run** (Ctrl+R) → choose pass 1 or 2.
For each word the script opens the right file, zooms to where the word is and waits. Drag across the word
(rules 4–5), then click **Save word**. Tick **Touches previous word** when there is no pause (rule 6).
The script snaps boundaries to zero crossings (rule 7), writes the label, and saves each finished file
into `kit/pass1` or `kit/pass2`. Close its pause window to stop; finished files are skipped next time.

## Manual way in Praat (about 45–60 min per pass)
1. Open Praat. *Open → Read from file…* → `corpus/<speaker>/<file>.wav`. Then *Open → Read from file…* →
   `hand_labels/kit/pass1/<file>.TextGrid` (an empty tier called `reference`).
2. Select both objects in the object list (Ctrl+click), then click **View & Edit**.
3. Go to the second given in `words_to_mark.txt` and find the word by listening.
4. Click at the word start on the **reference** tier (the top-right circle on the tier adds a boundary there),
   or click in the waveform and press **Enter**. Do the same at the word end.
5. Click inside the new interval and type the word exactly as listed (lower case).
6. Repeat for every listed word of the file, **in the listed order**. Leave everything else empty.
7. Save: *File → Save TextGrid as text file…* and **overwrite** `hand_labels/kit/pass1/<file>.TextGrid`.
8. Do the same for all 6 files. A day later, repeat with `hand_labels/kit/pass2/`.

## Then
```
python scripts/score_hand_labels.py      # -> analysis/hand_label_check.md, analysis/hand_label_errors.csv
```
The script checks that every listed word was labelled, in order. It reports:
* your own consistency (pass 1 vs. pass 2);
* each experiment's error against the mean of your two passes;
* a paired comparison against `arpa_base`.

## Record of the passes (filled in after marking)

| pass | tool | first word marked (UTC) | last word marked (UTC) | raw marks |
|---|---|---|---|---|
| 1 | browser tool | 2026-11-21 21:10 | 2026-11-21 22:16 | `kit/pass1_marks.json` |
| 2 | browser tool | 2026-11-22 07:09 | 2026-11-22 10:38 | `kit/pass2_marks.json` |

**Deviation from rule 2 (our own rule; the assignment sets none).** The two passes were separate sessions with a night's sleep between them, not the planned full day (exact times in the table above). Pass 2 was
done without looking at pass 1, and no marks were edited after scoring. A shorter gap can make the passes agree
more closely than fully independent passes would, so the reported pass-to-pass difference (median 11 ms) is, if
anything, an optimistic estimate of the reference's precision. Pass 2 lies on average 27 ms later than pass 1;
the reference uses the mean of the two passes.

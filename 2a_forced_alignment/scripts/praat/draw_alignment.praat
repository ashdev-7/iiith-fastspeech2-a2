# draw_alignment.praat
# Render waveform + spectrogram + MFA word/phone tiers for one file and save a PNG.
#
# Batch usage (from the repo root):
#   Praat.exe --run scripts/praat/draw_alignment.praat corpus/f2b/F2BJRLP1.wav outputs/arpa_base/f2b/F2BJRLP1.TextGrid analysis/figures/praat_F2BJRLP1.png 0 6
#   (the last two numbers are the time window in seconds; use 0 0 for the whole file)
#
# Interactive inspection: open the .wav and .TextGrid in Praat, select both,
# click "View & Edit" - boundaries can be checked against the spectrogram and audio.

form Draw alignment
    sentence wav_path corpus/f2b/F2BJRLP1.wav
    sentence textgrid_path outputs/arpa_base/f2b/F2BJRLP1.TextGrid
    sentence png_path analysis/figures/praat_F2BJRLP1.png
    real t_start 0
    real t_end 6
endform

sound = Read from file: wav_path$
tg = Read from file: textgrid_path$
if t_end <= t_start
    t_start = Get start time
    t_end = Get end time
endif

selectObject: sound
spec = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"

Erase all
# drawn at its printed size: the report's text width (6.6 in), 9-pt text
Font size: 9
Line width: 1

# waveform
Select outer viewport: 0, 6.6, 0, 0.65
selectObject: sound
Draw: t_start, t_end, 0, 0, "no", "Curve"

# spectrogram
Select outer viewport: 0, 6.6, 0.3, 1.95
selectObject: spec
Paint: t_start, t_end, 0, 5000, 100, "yes", 50, 6, 0, "no"
Marks left every: 1, 1000, "yes", "yes", "no"
Text left: "yes", "Frequency (Hz)"

# word + phone tiers with boundaries extended through the spectrogram
Select outer viewport: 0, 6.6, 0.3, 2.45
selectObject: tg
Draw: t_start, t_end, "yes", "yes", "yes"

Select outer viewport: 0, 6.6, 0, 2.45
# PNG from the Praat GUI / Praat --run; EPS when run headless (parselmouth has no bitmap renderer)
if endsWith (png_path$, ".eps")
    Save as EPS file: png_path$
else
    Save as 300-dpi PNG file: png_path$
endif
removeObject: sound, tg, spec

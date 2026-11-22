# Guided hand marking for hand_labels/ANNOTATION_PROTOCOL.md (one pass).
#
# Praat: Open > Open Praat script... > this file > Run (Ctrl+R). Choose the pass.
# For every listed word the script opens the right file, zooms to where the word is, and waits.
# You drag across the word (from its start to its end) and click "Save word". The script puts the
# boundaries on the nearest zero crossings, writes the label and, after the last word of a file,
# saves the TextGrid into kit/pass1 or kit/pass2. It never shows an aligner boundary.
# Files already finished are skipped, so you can stop (close the pause window) and run it again later.

form: "Hand marking (reference)"
    choice: "Pass", 1
        option: "1"
        option: "2"
endform

testMode = fileReadable: "TEST_MODE"
targets = Read Table from comma-separated file: "../targets.csv"
nrows = Get number of rows
writeInfoLine: "Hand marking, pass ", pass$
row = 1
while row <= nrows
    selectObject: targets
    file$ = Get value: row, "file"
    spk$ = Get value: row, "speaker"
    last = row
    more = 1
    while more and last < nrows
        nxt$ = Get value: last + 1, "file"
        if nxt$ = file$
            last = last + 1
        else
            more = 0
        endif
    endwhile
    nwords = last - row + 1
    tgPath$ = "pass" + pass$ + "/" + file$ + ".TextGrid"
    sound = Read from file: "../../corpus/" + spk$ + "/" + file$ + ".wav"
    dur = Get total duration
    tg = Read from file: tgPath$
    nlab = Count intervals where: 1, "is not equal to", ""
    if nlab = nwords
        appendInfoLine: file$, ": already done, skipped"
    else
        if nlab > 0
            removeObject: tg
            selectObject: sound
            tg = To TextGrid: "reference", ""
        endif
        fileDone = 0
        while not fileDone
            if not testMode
                selectObject: sound, tg
                View & Edit
            endif
            prevEnd = -1
            w = row
            while w <= last
                selectObject: targets
                word$ = Get value: w, "word"
                searchFrom = Get value: w, "search_from_s"
                a = max(0, searchFrom - 0.3)
                b = min(dur, searchFrom + 2.0)
                if left$(file$, 4) = "ISLE"
                    a = 0
                    b = dur
                endif
                touches_previous_word = 0
                if testMode
                    tStart = searchFrom + 0.05 + 0.3 * (w - row)
                    tEnd = tStart + 0.25
                else
                    editor: tg
                        Zoom: a, b
                    endeditor
                    beginPause: "Pass " + pass$ + ": " + file$ + ", word " + string$(w - row + 1) + " of " + string$(nwords)
                        comment: "Find the word   " + word$ + "   (it starts somewhere after " + string$(searchFrom) + " s)."
                        comment: "Drag across the whole word, from its start to its end. Tab = listen to the selection."
                        comment: "Start = first sign of the first sound: dark bands appear, hiss starts, or a stop burst (thin spike)."
                        comment: "End = last sign of the last sound. Include the burst of a final t, d, k, p, b or g."
                        comment: "To zoom in further: select a region, then Ctrl+N."
                        if prevEnd >= 0
                            boolean: "Touches previous word", 0
                            comment: "(tick it if there is no pause: the word then starts exactly where the last one ended)"
                        endif
                    clicked = endPause: "Save word", 1
                    editor: tg
                        tStart = Get start of selection
                        tEnd = Get end of selection
                    endeditor
                endif
                selectObject: sound
                if touches_previous_word
                    tStart = prevEnd
                else
                    tStart = Get nearest zero crossing: 1, tStart
                endif
                tEnd = Get nearest zero crossing: 1, tEnd
                problem$ = ""
                if tEnd - tStart < 0.02
                    problem$ = "The selection is too short. Drag across the whole word."
                elsif prevEnd >= 0 and tStart < prevEnd - 0.000001
                    problem$ = "The word starts before the end of the previous word. Select only this word (or tick Touches previous word)."
                endif
                if problem$ <> ""
                    if testMode
                        exitScript: problem$
                    endif
                    beginPause: "Please redo this word"
                        comment: problem$
                    clicked = endPause: "OK", 1
                else
                    selectObject: tg
                    if not touches_previous_word
                        @addBoundary: tStart
                    endif
                    @addBoundary: tEnd
                    i = Get interval at time: 1, (tStart + tEnd) / 2
                    Set interval text: 1, i, word$
                    prevEnd = tEnd
                    w = w + 1
                endif
            endwhile
            answer = 2
            if not testMode
                editor: tg
                    Zoom: max(0, a - 1), b
                endeditor
                beginPause: "Check " + file$
                    comment: "All " + string$(nwords) + " words of this file are marked. Look them over if you like."
                    comment: "Redo this file = clear its marks and start the file again."
                answer = endPause: "Redo this file", "Save and go on", 2
            endif
            if answer = 1
                removeObject: tg
                selectObject: sound
                tg = To TextGrid: "reference", ""
            else
                selectObject: tg
                Save as text file: tgPath$
                appendInfoLine: file$, ": saved (", nwords, " words)"
                fileDone = 1
            endif
        endwhile
    endif
    removeObject: sound, tg
    row = last + 1
endwhile
removeObject: targets
appendInfoLine: "Pass ", pass$, " finished. Tell Claude: pass ", pass$, " done."

procedure addBoundary: .t
    .i = Get interval at time: 1, .t
    .st = Get start time of interval: 1, .i
    if abs(.st - .t) > 0.000001
        Insert boundary: 1, .t
    endif
endproc

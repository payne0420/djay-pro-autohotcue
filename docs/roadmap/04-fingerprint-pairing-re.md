# 04 — Fingerprint pairing semantics: reverse-engineer and lift the guard

**Goal:** understand how djay actually pairs field names to values in
`mediaItemUserData` records that contain an `ADCAudioAlignmentFingerprint`
object, so the edit guard can be lifted and the ~72 affected tracks (2% of the
library — disproportionately the user's most-played ones) can get auto-cues
safely.

**Read first:** `docs/audio-alignment-fingerprint.md` — full forensics: record
anatomy, the June 2026 corruption incident, ranked hypotheses, and the five
reverse-engineering avenues. This brief is the execution plan for those
avenues; the background lives there and is not repeated here.

## Current state

- `apply` refuses any record whose parsed tree contains an Obj classname
  outside `djaydb.EDIT_SAFE_OBJ_CLASSNAMES` (guard added after the incident;
  real-library e2e asserts the skip on Move).
- Known: byte-exact round-trip holds for these records; our parser names the
  fingerprint Obj `timestampIdentifier`; records carry a duplicate root field
  name; the blob is zlib (5040 B decompressed; float encoding unidentified);
  djay scrambled the record after an in-place value edit.

## Plan (cheapest-first)

1. **Oracle experiment matrix** (no disassembly; the decisive one).
   On a *copy* of the library + real djay: mutate one variable at a time in a
   fingerprint record, open djay, load the track, quit, diff what djay
   re-saves. Variants: (a) byte-identical rewrite (control); (b) one cue time
   changed; (c) cue count changed; (d) beatGridEdits value changed; (e) c+d
   (the incident's combination). First scrambling variant isolates the
   trigger. Preserve every artifact (corrupted records to files *before* any
   restore — lesson learned). Caveat: djay must be pointed at the copy — check
   whether djay accepts a swapped `~/Music/djay/MediaLibrary.db` while signed
   out of iCloud sync, and disable sync during experiments if possible.
2. **Class-dump djay Pro.** `strings` / `otool -ov` / class-dump on the app
   binary for `ADCAudioAlignmentFingerprint`, `ADCMediaItemUserData`,
   `ADCBeatGridEdits`: property names and the TSAF (de)serializer methods.
   Tells us the true field keys and which class djay expects under
   `timestampIdentifier`. Legitimate interop RE on the user's own data.
3. **Corpus check.** Assert the structural invariant across all 72 records
   (script pattern in the fingerprint doc; 4/4 verified so far).
4. **Decode the blob** (optional, independent): try BE f32 / LE f64 / strided
   layouts; a real fingerprint plots as structure, not noise.
5. **tsaf grammar audit**: minimal synthetic blobs for each ambiguous token
   sequence (duplicate names, Marker-then-string, anonymous fields), validated
   against djay's rewrite via avenue 1.

## Acceptance criteria

- A written explanation of the pairing rule that predicts djay's rewrite
  byte-for-byte on at least the control + one mutated variant.
- Either: the guard is lifted (with a parser/serializer fix + regression tests
  + a successful real-djay load of an edited fingerprint record), or: a
  documented conclusion that editing these records requires regenerating the
  fingerprint (out of scope) and the guard stays.
- `tsaf.py` changes, if any, keep the full-library byte-exact round-trip test
  green — that invariant is non-negotiable.

## Risk notes

- All write experiments on copies only; djay quit before every swap.
- Do not run this concurrently with normal library use; iCloud sync can
  propagate corrupted records to other devices (sync is already known to be
  bypassed/flagged by external writes — see CLAUDE.md gotchas).

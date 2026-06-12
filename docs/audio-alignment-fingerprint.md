# ADCAudioAlignmentFingerprint — what we know, and the pairing problem

Working notes for reverse-engineering djay's audio-alignment fingerprint object and the
TSAF name↔value pairing semantics around it. Written after the June 2026 corruption
incident (below). Records containing this object are currently **refused by `apply`**
(classname allowlist guard in `cli._precheck_one`) — lifting that guard requires
settling the open questions here.

## What it is (observed)

- An Algoriddim object (`ADC*` class family) stored inside `mediaItemUserData` —
  the per-track record that holds user-owned, iCloud-synced state (cues, grid edits,
  rating, play count).
- Present on **72 of 3800** user-data records in the reference library (~2%) —
  plausibly the most-played / most-synced tracks.
- djay's accessor for it is `fingerprintFloatsArray` (from the crash selector), i.e.
  the payload decodes to a float array.
- Purpose (inferred from the name and its placement in synced user data): a compact
  acoustic signature used to **align time-based user data between different copies of
  the same song** (streaming match vs local file, different encodes/intros across
  synced devices). Cue/grid timestamps are file-relative; the fingerprint lets djay
  compute a per-version offset.

## Record anatomy (as parsed by tsaf today)

All 4 records inspected (3 random + Move) share the identical shape; field order as
our parser reports it:

```
('uuid')                 -> str
('userChangedCloudKeys') -> Arr        # contains the string "audioAlignmentFingerprint"
('titleIDs')             -> Arr
[('cuePoints')           -> Arr]       # when present
[('beatGridEdits')       -> Obj ADCBeatGridEdits]   # when present
('timestampIdentifier')  -> Obj ADCAudioAlignmentFingerprint
    ('timestampIdentifierData') -> Data, ~4.7 KB
    ('version')                 -> Marker 0x2e (= int 1)
('playCount')            -> Int | Marker
[('manualGain')          -> F32]
('timestampIdentifier')  -> Int        # duplicate root field name!
```

Raw-byte facts (consistent across all inspected records):

- `audioAlignmentFingerprint` appears **exactly once** in the bytes — inside the
  cloudKeys array. There is **no field named `audioAlignmentFingerprint`**.
- `timestampIdentifier` appears twice: once as the (parsed) name of the fingerprint
  Obj, once as the name of a trailing Int. `timestampIdentifierData` once (inside the
  Obj).
- The Data blob is **zlib-compressed** (`78 9c` header), length-prefixed.
  Move's: 4725 B compressed → **5040 B** decompressed = 1260×f32 or 630×f64.
  Plain little-endian f32 gives implausible values (±1e38) → **encoding open**;
  try big-endian f32, little-endian f64, or a (time, value) pair layout.
  djay calls it `fingerprintFloatsArray`, so floats are in there somehow.

## The June 2026 corruption incident

1. Baseline: Move's record (djay-canonical, 5613 B) contained cues + `beatGridEdits` +
   fingerprint and loaded fine in djay.
2. `apply` performed two in-place value replacements via `Obj.set`: new `cuePoints`
   array (6→8 cues) and new `beatGridEdits` value. Byte-exact round-trip held;
   the written record reconstructs to 5565 B with the fingerprint intact.
3. User opened djay, loaded Move → **"Could not load track"**,
   `-[ADCBeatGridEdits fingerprintFloatsArray]: unrecognized selector`.
4. djay re-saved the record as **761 B**: fingerprint object gone, the
   `ADCBeatGridEdits` object now sitting under the field name `timestampIdentifier`,
   `beatGridEdits` name token gone. Every subsequent load crashed (djay reads
   `timestampIdentifier`, expects the fingerprint class, calls
   `fingerprintFloatsArray` on our grid object).
5. Recovery: raw SQL `UPDATE` of the record bytes from the timestamped backup
   (no TSAF parsing involved). Track loads fine again.
6. **Forensics lesson: the 761 B scrambled record was overwritten during recovery
   without preserving a copy. Next time, dump the corrupted bytes to a file first.**
   What we recorded of it: fields `uuid, userChangedCloudKeys, titleIDs, cuePoints,
   timestampIdentifier(=ADCBeatGridEdits), playCount, manualGain,
   timestampIdentifier(Int)` — today's cues and anchor were preserved; only the
   fingerprint and the `beatGridEdits` name were lost.

The other 12 test tracks (no fingerprint) went through the identical write path and
load fine — djay even adopted our anchors and refined `fractionalBeatShift` itself.

## The pairing problem (hypotheses, ranked)

The crash + scramble prove djay binds these fields differently than our parser models,
at least when rewriting. Candidate explanations:

1. **Duplicate-name disambiguation.** djay genuinely keys the fingerprint Obj under
   `timestampIdentifier` (the cloud key `audioAlignmentFingerprint` being a different
   namespace), and disambiguates the two same-named root fields by order or by type.
   Our edits changed *relative positions/sizes* of content around them
   (cue count 6→8), possibly breaking djay's disambiguation on rewrite.
2. **Pairing frame-shift.** TSAF pairs `value` then `name`, names optional. If djay's
   true reading of this region differs from ours by one slot (e.g. the fingerprint Obj
   is anonymous and `timestampIdentifier` names the trailing Int), then any edit made
   through our model — even a pure value swap — can land in a slot djay reads
   differently. Note our model has no field named `audioAlignmentFingerprint` anywhere,
   yet djay's cloudKeys advertise one: *something* about our naming of this region is
   provably off.
3. **djay-side lazy-load bug.** djay may load the fingerprint lazily; our (valid)
   record changed sizes/offsets, the lazy load failed, and djay's writer serialized the
   record without the fingerprint, mis-writing the remaining fields. Under this theory
   our record was readable and the scramble is djay's writer bug — still triggered by
   our edit.

## Reverse-engineering avenues

1. **djay's writer as canonicalization oracle** (most promising, no disassembly).
   On a sacrificial library copy + the real djay app: mutate ONE thing at a time in a
   fingerprint record, open djay, load the track, quit, and diff what djay re-saves.
   Matrix: (a) byte-identical rewrite (control); (b) one cue time value changed;
   (c) cue count changed (the incident's delta); (d) beatGridEdits value changed;
   (e) c+d together (the incident). The first variant that djay scrambles isolates the
   trigger. djay's rewrite of healthy records also reveals its canonical field order
   and which names it re-emits — directly testing hypotheses 1 vs 2.
2. **Class-dump the app.** djay Pro is a Catalyst app under `/Applications`. `strings`
   / `otool -ov` / a class-dump tool on the main binary + frameworks for
   `ADCAudioAlignmentFingerprint`, `ADCMediaItemUserData`, `ADCBeatGridEdits`:
   property lists, ivar names, and the TSAF (de)serializer method names. The property
   names tell us the true field keys (is there a `timestampIdentifier` property on
   the user-data class AND on the fingerprint? what type does each expect?).
3. **Corpus diffing.** All 72 fingerprint records: assert the structural invariant
   holds library-wide (script in this repo's session history; 4/4 so far). Outliers =
   extra evidence about optional fields and ordering.
4. **Decode the blob.** Try BE f32, LE f64, strided layouts on the 5040 B payload;
   plot candidates — a real fingerprint should look like a smooth/structured curve,
   not noise. Knowing the content confirms what alignment djay does (and whether we
   could ever *regenerate* the object ourselves).
5. **tsaf grammar audit.** Targeted differential tests around the suspected zone:
   duplicate field names, Marker-valued fields followed by string tokens, and the
   anonymous-field rule (docstring: "if the token after a value is not a string token,
   the field is anonymous"). Construct minimal synthetic blobs for each ambiguous
   sequence and verify our pairing against djay's rewrite of the same bytes (via
   avenue 1).

## Safety rules in force

- `apply` refuses records whose parsed tree contains any Obj classname outside
  {user-data root, titleIDs, cue points, ADCBeatGridEdits} — skip, never edit.
- Byte-exact round-trip (`serialize(parse(b)) == b`) is **necessary but not
  sufficient** for edit safety: it proves reproduction, not that our name↔value
  pairing matches djay's. Do not weaken the guard based on round-trip evidence alone.
- Any experiment that writes near a fingerprint record happens on a **copy** of the
  library, and corrupted artifacts get preserved before restoring.

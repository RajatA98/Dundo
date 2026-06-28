# Suno Coaching Knowledge Base

> Expert reference injected into the `creatorAdvice` narrative prompt. The LLM uses
> this to coach a creator on improving the Suno/Udio track they uploaded. It is
> grounded knowledge, not fine-tuning — keep it current and concise.
>
> **Honesty (non-negotiable):** Suno metatags *guide, they do not guarantee* — never
> promise an exact output. Only reference the creator's own detected descriptors
> (tempo, key/mode, genre/mood tags). Never invent facts about their track. Suggest;
> don't overclaim.

## 1. Read the creator's intent, then tailor the advice

Most Suno creators fall into one of these. Infer the likely intent from the track's
descriptors and tone, and pitch the advice to it:

- **Personal / emotional song** (gifts, love songs, the TikTok "text-to-song" trend — the
  largest group, mostly non-musicians): they want it to *feel* personal and land
  emotionally. Coach toward dynamics, a memorable hook, and warm/human production —
  not technical jargon.
- **Content creator** (TikTok/Shorts/YouTube background, series themes): they want a
  catchy hook and a brand-fitting vibe, fast. Coach toward a strong 20-second hook,
  clear identity, loop-ability.
- **Jingle / commercial**: they want clean, on-brief, upbeat-and-simple. Coach toward
  clarity, one memorable line, tight length.
- **Songwriter / producer** (demos, style exploration): they want craft and
  distinctiveness. Coach toward arrangement contrast, a signature choice, and de-AI
  production moves.

Keep language plain for non-musicians; go deeper only if the track reads as crafted.

## 2. Suno's two-surface grammar (the core mental model)

- **Style field** = the *sound world*: genre lane, tempo feel, instrument palette,
  vocal type, atmosphere. Short, comma-separated descriptors (1–3 words each). The
  first 20–30 words matter most — top-load what matters.
- **Lyrics box** = *what happens inside that world*: section structure and local cues,
  via bracketed metatags.

## 3. Structure metatags (in the Lyrics box)

`[Intro] [Verse] [Pre-Chorus] [Chorus] [Bridge] [Outro]` — the spine.
Energy mechanics: `[Build-Up]` (rising tension), `[Drop]` (impact), `[Breakdown]`
(stripped-back space), `[Final Chorus]` (biggest hook return).
Place hard turns (`[Build-Up]`, `[Drop]`, `[Energy: High]`) *directly before* the
section they affect. One job per tag; don't stack conflicting instructions.

## 4. Delivery / descriptor / production tags

- Delivery: `[Whispered] [Belted] [Spoken] [Harmonized]`.
- Descriptor: `[Mood: …] [Energy: Low|Medium|High] [Vocal Style: …] [Instrument: …]`.
- Production words (in the Style field): "clear mix, present vocals, punchy drums,
  warm analog, studio quality, separated instruments." Remove "lo-fi/dusty" if you
  want clarity. Use a few *anchor* timbres, not an instrument shopping list.

## 5. "Resonate more" playbook (emotional impact)

1. **Dynamic contrast** — open the verse up (lower density, `[Energy: Low]`, one anchor
   timbre) so the chorus *lifts*. Put a `[Build-Up]` right before the `[Chorus]` and
   mark it `[Energy: High]`.
2. **A real hook** — one repeated line/melodic phrase; give its biggest return a
   `[Final Chorus]`.
3. **Intentional phrasing** — commas/hyphens in a lyric line force micro-pauses; use
   them where you want the listener to *feel* a word. Don't hyphenate randomly.
4. Match the move to the track: a slow, minor-key piece wants restraint and one
   swelling lift — not four competing sections.

## 6. "Stand out / sound less AI" playbook (distinctiveness)

1. **One signature choice the genre doesn't expect** — an off-genre instrument in a
   `[Breakdown]`, a half-time `[Bridge]`, an unusual vocal treatment.
2. **De-AI production** — add "warm analog, present vocals, clear mix, slight tape
   saturation"; this fights the generic-polish "AI" sound.
3. **Humanize** — small timing imperfections read as human; if editing, nudge phrases
   slightly off the grid.
4. **Swap in something real via Stems** — exporting stems and replacing the lead with a
   real take (or layering a live instrument) is the single biggest "de-AI" move.
5. Don't pitch an AI vocal more than ~2 semitones — artifacts magnify; re-generate in
   the right key instead.

## 7. Iteration moves (how to refine, not restart)

Re-roll a single section, **Extend** to grow the song, **Replace/Inpaint** a weak
section, edit lyrics for clarity, layer real elements via **Stems**. Prefer controlled
iteration over random full re-rolls.

## 8. Prompt-snippet patterns the coach can adapt

Adapt these to the creator's *detected* tempo / key / genre / mood — never paste generic.

- **Resonate (slow, minor, atmospheric):**
  Style field: `dream-pop, hazy, analog synth pads, intimate breathy vocal, ~70 BPM`
  Lyrics: `[Verse] [Energy: Low] … [Build-Up] [Chorus] [Energy: High] …`
- **Distinctive (mid-tempo, electronic/house):**
  Style field: `deep house, but swap the lead for a detuned mellotron, warm analog, live-feel hats`
  Lyrics: add a `[Breakdown]` with one unexpected acoustic instrument, or a half-time `[Bridge]`.
- **Catchy hook (content/TikTok):**
  "Front-load the hook: put your strongest line in the first `[Chorus]` and again as a
  `[Final Chorus]`; Style field: `upbeat indie-pop, bright, punchy drums, big catchy hook`."
- **Less-AI (any):**
  "Add `present vocals, clear mix, warm analog` to the Style field; if you can, export
  Stems and re-record the lead line for a human take."

## 9. What the coach should output

Given the track's descriptors, return concrete, copyable advice — typically **one
"make it resonate more" idea and one "make it more distinctive" idea**, each with a
short, adaptable Suno snippet (a Style-field line and/or the relevant structure
metatags). Ground every suggestion in the detected descriptors. Stay warm and plain.
End with the honest reminder that Suno *guides* the output — it may take a couple of
re-rolls to land.

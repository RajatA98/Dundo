/**
 * SunoPill — a small "likely suno" pill rendered inside the AI Music Detector row
 * when the response's `likely_source === "suno"`.
 *
 * Phase 3 scaffold. The visual contract is locked in:
 *   factory/artifacts/ui_mockup_v2_suno_flare.html (.suno-pill, .suno-pill .sigil)
 *
 * Suno-flare tokens (add to index.css @theme block — see CODEX_PHASE_3.md):
 *   --color-suno: #F25C54
 *   --color-suno-soft: rgba(242, 92, 84, 0.10)
 *   --color-suno-deep: #B8403A
 *
 * Reserved for the AI Music Detector likely-Suno case ONLY. Never used as
 * a primary brand color anywhere else on the page. (See LOCKED_DECISIONS
 * "ACRCloud — AI Music Detector" section + the design prompt's "Suno flare"
 * addendum.)
 *
 * @param {Object} props
 * @param {string} [props.label="likely suno"] - the pill text (lowercase per Suno brand)
 */
export default function SunoPill({ label = 'likely suno' }) {
  // The schematic S-curve sigil — echoes Suno's wonk mark without imitating it.
  // Stroke uses --color-suno-deep so the sigil reads inside the soft-rose pill.
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-mono text-[11px] font-semibold uppercase leading-none"
      style={{
        background: 'var(--color-suno-soft)',
        color: 'var(--color-suno-deep)',
        borderColor: 'rgba(242, 92, 84, 0.22)',
        letterSpacing: '0.04em',
        textTransform: 'lowercase', // brand is lowercase
      }}
    >
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
        <path
          d="M7.4 2.5C6.6 1.7 5.4 1.6 4.6 2.2C3.6 3 3.6 4.5 4.7 5.1L6.1 5.9C7 6.4 7 7.6 6.1 8C5.4 8.4 4.4 8.2 3.8 7.4"
          stroke="var(--color-suno-deep)"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
      </svg>
      {label}
    </span>
  )
}

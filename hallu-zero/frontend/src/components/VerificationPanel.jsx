import React, { useState } from 'react'
import clsx from 'clsx'

const VERDICT_COLOR = {
  supported:    { color: '#00ff88', bg: 'rgba(0,255,136,0.08)', label: 'SUP' },
  unsupported:  { color: '#ffb347', bg: 'rgba(255,179,71,0.08)', label: 'UNS' },
  contradicted: { color: '#ff4466', bg: 'rgba(255,68,102,0.08)', label: 'CON' },
  unverifiable: { color: '#5a5a7a', bg: 'rgba(90,90,122,0.12)', label: 'UNV' },
}

function ClaimRow({ claim }) {
  const v = VERDICT_COLOR[claim.verdict] || VERDICT_COLOR.unverifiable
  return (
    <div
      className="flex gap-3 py-2 px-3 rounded"
      style={{ background: v.bg, borderLeft: `2px solid ${v.color}` }}
    >
      <span
        className="text-xs font-bold mt-0.5 flex-shrink-0 tracking-widest"
        style={{ color: v.color }}
      >
        {v.label}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-ink-200 text-xs leading-relaxed">{claim.claim}</p>
        {claim.evidence && (
          <p className="text-ink-400 text-xs mt-1 italic truncate">
            ↳ {claim.evidence}
          </p>
        )}
      </div>
      <span className="text-ink-400 text-xs flex-shrink-0">
        {Math.round(claim.confidence * 100)}%
      </span>
    </div>
  )
}

export function VerificationPanel({ verification }) {
  const [open, setOpen] = useState(false)

  if (!verification || Object.keys(verification).length === 0) return null

  const claims = verification.claim_verdicts || []
  const criticScore = Math.round((verification.critic_score || 0) * 100)
  const overall = Math.round((verification.overall_confidence || 0) * 100)
  const passed = verification.passed

  return (
    <div className="mt-3 border border-ink-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-ink-800 hover:bg-ink-700 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span
            className="text-xs font-bold tracking-widest"
            style={{ color: passed ? '#00ff88' : '#ff4466' }}
          >
            {passed ? '● VERIFIED' : '● UNVERIFIED'}
          </span>
          <span className="text-ink-400 text-xs">
            {claims.length} claims · critic {criticScore}% · overall {overall}%
          </span>
        </div>
        <span className="text-ink-400 text-xs">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="px-4 py-3 space-y-2 bg-ink-900">
          {/* Scores bar */}
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div>
              <div className="text-ink-400 text-xs mb-1">Claim score</div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${overall}%`,
                    background: overall >= 80 ? '#00ff88' : overall >= 60 ? '#ffb347' : '#ff4466',
                  }}
                />
              </div>
            </div>
            <div>
              <div className="text-ink-400 text-xs mb-1">Critic score</div>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${criticScore}%`,
                    background: criticScore >= 80 ? '#00ff88' : criticScore >= 60 ? '#ffb347' : '#ff4466',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Critic feedback */}
          {verification.critic_feedback && (
            <p className="text-ink-300 text-xs px-3 py-2 bg-ink-800 rounded italic">
              Critic: {verification.critic_feedback}
            </p>
          )}

          {/* Claims */}
          {claims.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-ink-500 text-xs tracking-widest uppercase mb-2">
                Atomic claims
              </div>
              {claims.map((c, i) => <ClaimRow key={i} claim={c} />)}
            </div>
          )}

          {verification.regeneration_hint && (
            <p className="text-xs text-signal-amber px-3 py-2 bg-ink-800 rounded">
              ↻ Hint: {verification.regeneration_hint}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

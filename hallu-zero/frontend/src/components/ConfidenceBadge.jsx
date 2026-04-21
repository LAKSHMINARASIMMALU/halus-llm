import React from 'react'
import clsx from 'clsx'

export function ConfidenceBadge({ score, passed, size = 'md' }) {
  const pct = Math.round((score || 0) * 100)

  const color =
    pct >= 80 ? '#00ff88' :
    pct >= 60 ? '#ffb347' :
    '#ff4466'

  const label =
    pct >= 80 ? 'HIGH' :
    pct >= 60 ? 'MED' :
    'LOW'

  return (
    <div className={clsx('flex items-center gap-2', size === 'sm' ? 'text-xs' : 'text-xs')}>
      <span
        className="signal-dot"
        style={{ background: color, boxShadow: `0 0 6px ${color}60` }}
      />
      <span style={{ color }} className="font-medium tracking-widest">
        {label} {pct}%
      </span>
      {passed !== undefined && (
        <span className="text-ink-400 ml-1">
          {passed ? '✓ verified' : '⚠ unverified'}
        </span>
      )}
    </div>
  )
}

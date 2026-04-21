import React, { useState } from 'react'

export function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false)
  const [expanded, setExpanded] = useState(null)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-2 border border-ink-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2 bg-ink-800 hover:bg-ink-700 transition-colors text-left"
      >
        <span className="text-ink-300 text-xs tracking-widest">
          ◈ {sources.length} SOURCE{sources.length !== 1 ? 'S' : ''}
        </span>
        <span className="text-ink-500 text-xs">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="divide-y divide-ink-800">
          {sources.map((src, i) => (
            <div key={i} className="bg-ink-900">
              <button
                onClick={() => setExpanded(expanded === i ? null : i)}
                className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-ink-800 transition-colors text-left"
              >
                <span className="text-ink-500 text-xs w-5 text-center">{i + 1}</span>
                <span className="flex-1 text-ink-200 text-xs truncate">{src.source}</span>
                <span
                  className="text-xs font-mono flex-shrink-0"
                  style={{ color: src.score > 0.7 ? '#00ff88' : src.score > 0.4 ? '#ffb347' : '#5a5a7a' }}
                >
                  {(src.score * 100).toFixed(0)}%
                </span>
                <span className="text-ink-500 text-xs">{expanded === i ? '−' : '+'}</span>
              </button>

              {expanded === i && (
                <div className="px-4 pb-3">
                  <p className="text-ink-300 text-xs leading-relaxed bg-ink-800 p-3 rounded border-l-2 border-signal-blue">
                    {src.content}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

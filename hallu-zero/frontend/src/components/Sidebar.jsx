import React from 'react'
import { NavLink } from 'react-router-dom'
import clsx from 'clsx'

const NAV = [
  { to: '/',         icon: '◆', label: 'Chat' },
  { to: '/ingest',   icon: '◈', label: 'Ingest' },
  { to: '/dashboard',icon: '◎', label: 'Dashboard' },
  { to: '/training', icon: '▲', label: 'Training' },
]

export function Sidebar({ health }) {
  const isOnline = health?.status === 'ok'

  return (
    <aside className="w-56 flex-shrink-0 flex flex-col glass border-r border-ink-800 h-screen sticky top-0">
      {/* Logo */}
      <div className="px-6 py-5 border-b border-ink-800">
        <h1 className="font-display text-xl text-ink-50 tracking-tight leading-none">
          Hallu<span style={{ color: '#00ff88' }}>Zero</span>
        </h1>
        <p className="text-ink-500 text-xs mt-1">anti-hallucination stack</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all',
              isActive
                ? 'bg-ink-700 text-ink-50 border border-ink-600'
                : 'text-ink-400 hover:text-ink-200 hover:bg-ink-800'
            )}
          >
            <span className="text-xs">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Status */}
      <div className="px-4 py-4 border-t border-ink-800">
        <div className="flex items-center gap-2 mb-2">
          <span
            className="signal-dot"
            style={{
              background: isOnline ? '#00ff88' : '#ff4466',
              boxShadow: isOnline ? '0 0 6px #00ff8860' : 'none',
            }}
          />
          <span className="text-xs text-ink-400">
            {isOnline ? 'Ollama online' : 'Ollama offline'}
          </span>
        </div>
        {health?.config && (
          <div className="space-y-1">
            <div className="text-xs text-ink-600 truncate">
              LLM: {health.config.llm_model}
            </div>
            <div className="text-xs text-ink-600 truncate">
              Docs: {health.rag?.vector_store_docs ?? '—'}
            </div>
            <div className="text-xs text-ink-600 truncate">
              Threshold: {Math.round((health.config.confidence_threshold || 0) * 100)}%
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

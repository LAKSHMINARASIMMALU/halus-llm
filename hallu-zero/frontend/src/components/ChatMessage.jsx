import React from 'react'
import ReactMarkdown from 'react-markdown'
import { ConfidenceBadge } from './ConfidenceBadge'
import { VerificationPanel } from './VerificationPanel'
import { SourcesPanel } from './SourcesPanel'
import clsx from 'clsx'

function ThumbButton({ onClick, active, positive }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'text-xs px-2 py-1 rounded border transition-all',
        active
          ? positive
            ? 'border-signal-green text-signal-green bg-signal-green/10'
            : 'border-signal-red text-signal-red bg-signal-red/10'
          : 'border-ink-700 text-ink-500 hover:border-ink-500 hover:text-ink-300'
      )}
    >
      {positive ? '▲' : '▼'}
    </button>
  )
}

export function ChatMessage({ message, onFeedback }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end animate-slide-up">
        <div className="max-w-[75%] px-4 py-3 rounded-xl rounded-tr-sm bg-ink-700 border border-ink-600">
          <p className="text-ink-100 text-sm leading-relaxed">{message.content}</p>
          <p className="text-ink-500 text-xs mt-1.5 text-right">
            {new Date(message.timestamp).toLocaleTimeString()}
          </p>
        </div>
      </div>
    )
  }

  if (message.role === 'error') {
    return (
      <div className="flex animate-slide-up">
        <div className="max-w-[85%] px-4 py-3 rounded-xl rounded-tl-sm bg-signal-red/10 border border-signal-red/30">
          <p className="text-signal-red text-xs font-mono">⚠ {message.content}</p>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex animate-slide-up">
      <div className="max-w-[90%] w-full">
        {/* Header bar */}
        <div className="flex items-center gap-3 mb-2">
          <span className="text-ink-600 text-xs">
            ◆ {message.model || 'assistant'}
          </span>
          <span
            className="text-xs px-2 py-0.5 rounded-full border"
            style={{
              color: message.mode === 'rag' ? '#00ff88' : '#4488ff',
              borderColor: message.mode === 'rag' ? '#00ff8840' : '#4488ff40',
              background: message.mode === 'rag' ? '#00ff8810' : '#4488ff10',
            }}
          >
            {message.mode === 'rag' ? '◈ RAG' : '◆ Direct'}
          </span>
          {message.attempts > 1 && (
            <span className="text-ink-500 text-xs">
              ↻ {message.attempts} attempts
            </span>
          )}
          <div className="flex-1" />
          <ConfidenceBadge
            score={message.confidence}
            passed={message.verification?.passed}
          />
        </div>

        {/* Response content */}
        <div className="glass px-5 py-4 rounded-xl rounded-tl-sm">
          <div className="prose-dark text-sm leading-relaxed">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>

          {/* Sources */}
          <SourcesPanel sources={message.sources} />

          {/* Verification */}
          <VerificationPanel verification={message.verification} />

          {/* Footer: feedback + timestamp */}
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-ink-800">
            <p className="text-ink-600 text-xs">
              {new Date(message.timestamp).toLocaleTimeString()}
            </p>

            {!message.feedbackGiven ? (
              <div className="flex items-center gap-2">
                <span className="text-ink-600 text-xs">Was this accurate?</span>
                <ThumbButton
                  onClick={() => onFeedback(message, 1)}
                  positive={true}
                  active={message.feedbackGiven === 1}
                />
                <ThumbButton
                  onClick={() => onFeedback(message, -1)}
                  positive={false}
                  active={message.feedbackGiven === -1}
                />
              </div>
            ) : (
              <span className="text-ink-500 text-xs italic">
                {message.feedbackGiven === 1 ? '▲ marked helpful' : '▼ marked unhelpful'}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

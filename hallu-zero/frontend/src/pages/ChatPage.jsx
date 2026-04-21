import React, { useRef, useEffect, useState } from 'react'
import { useChat } from '../hooks/useChat'
import { ChatMessage } from '../components/ChatMessage'
import { chatApi } from '../lib/api'

const SUGGESTIONS = [
  'What is machine learning?',
  'Explain neural networks simply.',
  'What is the difference between AI and ML?',
  'How does a large language model work?',
]

const MODE_OPTIONS = [
  { value: 'auto',   label: 'Auto',   icon: '◎', desc: 'RAG if docs exist, else Direct' },
  { value: 'direct', label: 'Direct', icon: '◆', desc: 'Ollama knowledge only' },
  { value: 'rag',    label: 'RAG',    icon: '◈', desc: 'Your documents only' },
]

export function ChatPage() {
  const { messages, loading, mode, setMode, sendMessage, submitFeedback, clearMessages } = useChat()
  const [input, setInput] = useState('')
  const [clearing, setClearing] = useState(false)
  const [clearMsg, setClearMsg] = useState('')
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = '48px'
    }
    await sendMessage(q)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClearKnowledge = async () => {
    if (!window.confirm('Clear all ingested documents from the knowledge base?')) return
    setClearing(true)
    try {
      await chatApi.clearKnowledge()
      setClearMsg('Knowledge base cleared!')
      setTimeout(() => setClearMsg(''), 3000)
    } catch (e) {
      setClearMsg('Error clearing: ' + e.message)
    } finally {
      setClearing(false)
    }
  }

  const currentMode = MODE_OPTIONS.find(m => m.value === mode)
  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col h-screen">
      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-ink-800 bg-ink-950/80 backdrop-blur flex-shrink-0 gap-4">
        {/* Mode toggle */}
        <div className="flex items-center gap-1 bg-ink-800 rounded-lg p-1">
          {MODE_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setMode(opt.value)}
              title={opt.desc}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs transition-all"
              style={{
                background: mode === opt.value ? (
                  opt.value === 'rag' ? '#00ff8820' :
                  opt.value === 'direct' ? '#4488ff20' : '#3a3a52'
                ) : 'transparent',
                color: mode === opt.value ? (
                  opt.value === 'rag' ? '#00ff88' :
                  opt.value === 'direct' ? '#4488ff' : '#d8d8ee'
                ) : '#5a5a7a',
                border: mode === opt.value ? (
                  opt.value === 'rag' ? '1px solid #00ff8840' :
                  opt.value === 'direct' ? '1px solid #4488ff40' : '1px solid #3a3a52'
                ) : '1px solid transparent',
              }}
            >
              <span>{opt.icon}</span>
              <span>{opt.label}</span>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 text-xs text-ink-500">
          {clearMsg && (
            <span style={{ color: clearMsg.startsWith('Error') ? '#ff4466' : '#00ff88' }}>
              {clearMsg}
            </span>
          )}
          <button
            onClick={handleClearKnowledge}
            disabled={clearing}
            className="hover:text-signal-red transition-colors"
            title="Clear all ingested documents"
          >
            {clearing ? '◌ clearing...' : '⊘ clear knowledge'}
          </button>
          {messages.length > 0 && (
            <button onClick={clearMessages} className="hover:text-ink-300 transition-colors">
              ✕ clear chat
            </button>
          )}
        </div>
      </div>

      {/* Mode description bar */}
      <div
        className="px-6 py-1.5 text-xs border-b border-ink-900 flex items-center gap-2"
        style={{
          color: mode === 'rag' ? '#00ff8880' : mode === 'direct' ? '#4488ff80' : '#5a5a7a',
          background: mode === 'rag' ? '#00ff8806' : mode === 'direct' ? '#4488ff06' : 'transparent',
        }}
      >
        <span>{currentMode?.icon}</span>
        <span>{currentMode?.desc}</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6 grid-bg">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center h-full text-center py-16 animate-fade-in">
            <div className="text-5xl mb-4 opacity-20 font-display">◆</div>
            <h2 className="font-display text-2xl text-ink-200 mb-2">Ask anything</h2>
            <p className="text-ink-500 text-sm max-w-sm mb-2">
              {mode === 'direct'
                ? 'Direct mode — answers from Ollama\'s training knowledge.'
                : mode === 'rag'
                ? 'RAG mode — answers from your ingested documents only.'
                : 'Auto mode — uses your documents if available, otherwise Ollama knowledge.'}
            </p>
            <div className="grid grid-cols-1 gap-2 w-full max-w-md mt-6">
              {SUGGESTIONS.map((s, i) => (
                <button key={i} onClick={() => sendMessage(s)}
                  className="text-left text-xs text-ink-400 px-4 py-2.5 rounded-lg border border-ink-800 hover:border-ink-600 hover:text-ink-200 hover:bg-ink-800 transition-all">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <ChatMessage key={msg.id} message={msg} onFeedback={submitFeedback} />
        ))}

        {loading && (
          <div className="flex animate-fade-in">
            <div className="glass px-5 py-4 rounded-xl rounded-tl-sm">
              <div className="flex items-center gap-3 text-ink-400 text-xs">
                <span className="animate-spin">◌</span>
                <span>
                  {mode === 'direct' ? 'Thinking...' : 'Retrieving · Generating · Verifying...'}
                </span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-ink-800 bg-ink-950/90 backdrop-blur flex-shrink-0">
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={
              mode === 'direct' ? 'Ask anything — powered by Ollama...' :
              mode === 'rag' ? 'Ask about your documents...' :
              'Ask anything...'
            }
            rows={1}
            className="flex-1 bg-ink-800 border border-ink-700 rounded-xl px-4 py-3 text-sm text-ink-100 placeholder-ink-600 resize-none focus:outline-none focus:border-ink-500 transition-colors"
            style={{ minHeight: '48px', maxHeight: '160px' }}
            onInput={e => {
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
            }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="px-5 py-3 rounded-xl text-sm font-medium transition-all flex-shrink-0"
            style={{
              background: input.trim() && !loading ? (
                mode === 'rag' ? '#00ff88' : mode === 'direct' ? '#4488ff' : '#00ff88'
              ) : '#1e1e2e',
              color: input.trim() && !loading ? '#0a0a0f' : '#5a5a7a',
              cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
            }}
          >
            {loading ? '◌' : '◆'}
          </button>
        </div>
        <p className="text-ink-700 text-xs mt-2 text-center">
          Mode: {currentMode?.label} · RLHF feedback loop active
        </p>
      </div>
    </div>
  )
}

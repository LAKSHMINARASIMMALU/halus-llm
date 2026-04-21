import React, { useEffect, useState } from 'react'
import { chatApi } from '../lib/api'

function StatCard({ label, value, color, sub }) {
  return (
    <div className="glass rounded-xl p-5">
      <div className="text-ink-500 text-xs tracking-widest uppercase mb-2">{label}</div>
      <div className="text-2xl font-display" style={{ color: color || '#d8d8ee' }}>
        {value}
      </div>
      {sub && <div className="text-ink-500 text-xs mt-1">{sub}</div>}
    </div>
  )
}

function FeedbackRow({ item }) {
  const color = item.rating === 1 ? '#00ff88' : '#ff4466'
  return (
    <div className="flex items-center gap-3 px-4 py-2.5 border-b border-ink-800 last:border-0">
      <span className="text-xs font-bold" style={{ color }}>
        {item.rating === 1 ? '▲' : '▼'}
      </span>
      <span className="flex-1 text-ink-300 text-xs truncate">{item.query}</span>
      <span
        className="text-xs flex-shrink-0"
        style={{ color: item.confidence_score > 0.65 ? '#00ff88' : '#ff4466' }}
      >
        {Math.round(item.confidence_score * 100)}%
      </span>
      <span className="text-ink-600 text-xs flex-shrink-0">
        {item.created_at ? new Date(item.created_at).toLocaleTimeString() : ''}
      </span>
    </div>
  )
}

export function DashboardPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const stats = await chatApi.stats()
      setData(stats)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const fb = data?.feedback || {}
  const rag = data?.rag || {}

  return (
    <div className="flex flex-col h-screen">
      <div className="flex items-center justify-between px-6 py-3 border-b border-ink-800 bg-ink-950/80 backdrop-blur flex-shrink-0">
        <span className="text-ink-500 text-xs tracking-widest uppercase">Dashboard</span>
        <button
          onClick={load}
          className="text-ink-500 hover:text-signal-green text-xs transition-colors"
        >
          ↻ refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8 grid-bg">
        {loading ? (
          <div className="flex items-center justify-center h-40 text-ink-500 text-sm">
            <span className="animate-spin mr-2">◌</span> Loading stats...
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-8">

            {/* Feedback metrics */}
            <div>
              <h2 className="font-display text-xl text-ink-200 mb-4">Feedback & RLHF</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                  label="Total feedback"
                  value={fb.total_feedback ?? 0}
                />
                <StatCard
                  label="Satisfaction"
                  value={`${Math.round((fb.satisfaction_rate || 0) * 100)}%`}
                  color={fb.satisfaction_rate > 0.7 ? '#00ff88' : '#ffb347'}
                  sub={`${fb.positives ?? 0}↑ ${fb.negatives ?? 0}↓`}
                />
                <StatCard
                  label="Avg confidence"
                  value={`${Math.round((fb.avg_confidence || 0) * 100)}%`}
                  color={fb.avg_confidence > 0.65 ? '#00ff88' : '#ff4466'}
                />
                <StatCard
                  label="Verification pass"
                  value={`${Math.round((fb.verification_pass_rate || 0) * 100)}%`}
                  color="#4488ff"
                />
              </div>
            </div>

            {/* RAG metrics */}
            <div>
              <h2 className="font-display text-xl text-ink-200 mb-4">Knowledge Base</h2>
              <div className="grid grid-cols-2 gap-4">
                <StatCard
                  label="Vector store chunks"
                  value={rag.vector_store_docs ?? 0}
                  color="#aa66ff"
                  sub="ChromaDB"
                />
                <StatCard
                  label="BM25 index size"
                  value={rag.bm25_docs ?? 0}
                  color="#4488ff"
                  sub="Sparse index"
                />
              </div>
            </div>

            {/* Training pairs */}
            <div className="glass rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-ink-200 text-sm font-medium">Training pairs (DPO)</h3>
                <span className="text-signal-purple text-sm font-mono">
                  {fb.training_pairs_available ?? 0} pairs
                </span>
              </div>
              <p className="text-ink-400 text-xs mb-3">
                Positive/negative feedback pairs are automatically matched to create DPO training data.
                Export them from the Training page.
              </p>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{
                    width: `${Math.min(100, ((fb.training_pairs_available || 0) / 100) * 100)}%`,
                    background: '#aa66ff',
                  }}
                />
              </div>
              <p className="text-ink-600 text-xs mt-1">
                {fb.training_pairs_available ?? 0} / 100 pairs to recommended fine-tune threshold
              </p>
            </div>

            {/* Recent feedback */}
            {data?.recent_feedback?.length > 0 && (
              <div>
                <h2 className="font-display text-xl text-ink-200 mb-4">Recent Feedback</h2>
                <div className="glass rounded-xl overflow-hidden">
                  {data.recent_feedback.map((item, i) => (
                    <FeedbackRow key={i} item={item} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

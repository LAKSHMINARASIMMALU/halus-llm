import React, { useState } from 'react'
import { chatApi } from '../lib/api'

export function TrainingPage() {
  const [pairs, setPairs] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchPairs = async () => {
    setLoading(true)
    try {
      const data = await chatApi.trainingPairs()
      setPairs(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const downloadJson = () => {
    if (!pairs) return
    const blob = new Blob([JSON.stringify(pairs.pairs, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `hallu_zero_training_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="flex items-center px-6 py-3 border-b border-ink-800 bg-ink-950/80 backdrop-blur flex-shrink-0">
        <span className="text-ink-500 text-xs tracking-widest uppercase">Training Data</span>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8 grid-bg">
        <div className="max-w-2xl mx-auto space-y-8">
          <div>
            <h2 className="font-display text-2xl text-ink-100 mb-1">RLHF Training Export</h2>
            <p className="text-ink-400 text-sm">
              Matched positive/negative feedback pairs formatted for{' '}
              <span className="text-signal-purple">DPO (Direct Preference Optimization)</span> fine-tuning.
            </p>
          </div>

          {/* Pipeline explanation */}
          <div className="glass rounded-xl p-5 space-y-3">
            <h3 className="text-ink-200 text-sm font-medium">How the RLHF loop works</h3>
            {[
              ['1. Collect feedback', 'Users rate responses ▲/▼ on every assistant message'],
              ['2. Match pairs', 'Positive + negative responses to similar queries are paired automatically'],
              ['3. Export DPO data', 'Pairs exported as (chosen, rejected) JSON for fine-tuning'],
              ['4. Fine-tune', 'Use with trl / unsloth / axolotl on local GPU to improve the base model'],
            ].map(([step, desc]) => (
              <div key={step} className="flex gap-3">
                <span className="text-signal-purple text-xs font-mono flex-shrink-0 w-32">{step}</span>
                <span className="text-ink-400 text-xs">{desc}</span>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={fetchPairs}
              disabled={loading}
              className="px-5 py-2.5 rounded-lg text-sm border border-ink-600 text-ink-300 hover:border-ink-400 hover:text-ink-100 transition-colors disabled:opacity-50"
            >
              {loading ? '◌ Loading...' : '↓ Fetch pairs'}
            </button>
            {pairs && pairs.count > 0 && (
              <button
                onClick={downloadJson}
                className="px-5 py-2.5 rounded-lg text-sm font-medium transition-colors"
                style={{ background: '#00ff88', color: '#0a0a0f' }}
              >
                ↓ Download JSON ({pairs.count} pairs)
              </button>
            )}
          </div>

          {/* Pairs preview */}
          {pairs && (
            <div>
              {pairs.count === 0 ? (
                <div className="text-ink-500 text-sm text-center py-8">
                  No training pairs yet. Submit more feedback (▲/▼) to generate pairs.
                </div>
              ) : (
                <div className="space-y-4">
                  <p className="text-ink-400 text-xs">{pairs.count} pairs · format: {pairs.format}</p>
                  {pairs.pairs.slice(0, 3).map((p, i) => (
                    <div key={i} className="glass rounded-xl overflow-hidden">
                      <div className="px-4 py-2 bg-ink-800 border-b border-ink-700">
                        <span className="text-ink-400 text-xs">Prompt:</span>
                        <span className="text-ink-200 text-xs ml-2">{p.prompt?.slice(0, 80)}...</span>
                      </div>
                      <div className="p-4 grid grid-cols-2 gap-3">
                        <div>
                          <div className="text-signal-green text-xs mb-1 tracking-widest">CHOSEN</div>
                          <p className="text-ink-300 text-xs line-clamp-3">{p.chosen?.slice(0, 150)}...</p>
                        </div>
                        <div>
                          <div className="text-signal-red text-xs mb-1 tracking-widest">REJECTED</div>
                          <p className="text-ink-300 text-xs line-clamp-3">{p.rejected?.slice(0, 150)}...</p>
                        </div>
                      </div>
                      <div className="px-4 py-2 border-t border-ink-800 text-xs text-ink-500">
                        reward gap: {p.reward_gap?.toFixed(3)}
                      </div>
                    </div>
                  ))}
                  {pairs.count > 3 && (
                    <p className="text-ink-500 text-xs text-center">
                      +{pairs.count - 3} more pairs in downloaded file
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Fine-tuning command */}
          <div className="glass rounded-xl p-5">
            <h3 className="text-ink-300 text-sm font-medium mb-3">Fine-tune with trl (DPO)</h3>
            <pre className="bg-ink-900 border border-ink-800 rounded-lg p-4 text-xs text-signal-green overflow-x-auto">
{`pip install trl transformers datasets

# Quick DPO fine-tune (4-bit QLoRA)
python -c "
from trl import DPOTrainer
from transformers import AutoModelForCausalLM
import json

with open('hallu_zero_training_*.json') as f:
    pairs = json.load(f)

# Feed pairs to DPOTrainer...
# See: huggingface.co/docs/trl/dpo_trainer
"`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  )
}

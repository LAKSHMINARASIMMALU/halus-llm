import React, { useState } from 'react'
import { UploadZone } from '../components/UploadZone'

export function IngestPage() {
  const [history, setHistory] = useState([])

  const handleIngested = (result) => {
    setHistory(h => [{
      ...result,
      timestamp: new Date().toLocaleTimeString(),
    }, ...h])
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="flex items-center px-6 py-3 border-b border-ink-800 bg-ink-950/80 backdrop-blur flex-shrink-0">
        <span className="text-ink-500 text-xs tracking-widest uppercase">Ingest Documents</span>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8 grid-bg">
        <div className="max-w-xl mx-auto space-y-8">
          <div>
            <h2 className="font-display text-2xl text-ink-100 mb-1">
              Add to Knowledge Base
            </h2>
            <p className="text-ink-400 text-sm">
              Documents are chunked, embedded via <span className="text-signal-green">nomic-embed-text</span>,
              and indexed into ChromaDB + BM25 for multi-stage retrieval.
            </p>
          </div>

          <UploadZone onIngested={handleIngested} />

          <div className="glass rounded-xl p-5">
            <h3 className="text-ink-300 text-sm font-medium mb-3">
              Or ingest from the CLI
            </h3>
            <pre className="bg-ink-900 border border-ink-800 rounded-lg p-4 text-xs text-signal-green overflow-x-auto">
{`# Drop files into backend/data/documents/ then:
cd backend
python -m app.rag.ingest --path ./data/documents`}
            </pre>
          </div>

          {history.length > 0 && (
            <div>
              <h3 className="text-ink-400 text-xs tracking-widest uppercase mb-3">
                Ingestion history
              </h3>
              <div className="space-y-2">
                {history.map((h, i) => (
                  <div key={i} className="flex items-center justify-between px-4 py-2.5 rounded-lg bg-ink-800 border border-ink-700">
                    <div className="text-xs text-ink-300">
                      {h.files_processed} file(s) · {h.chunks_ingested} chunks
                    </div>
                    <div className="text-xs text-ink-500">{h.timestamp}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

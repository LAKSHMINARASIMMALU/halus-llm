import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { chatApi } from '../lib/api'
import clsx from 'clsx'

export function UploadZone({ onIngested }) {
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback(async (files) => {
    if (!files.length) return
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const res = await chatApi.ingest(files)
      setResult(res)
      onIngested?.(res)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setUploading(false)
    }
  }, [onIngested])

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/pdf': ['.pdf'],
      'application/json': ['.json'],
    },
    maxFiles: 20,
  })

  return (
    <div className="space-y-3">
      <div
        {...getRootProps()}
        className={clsx(
          'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all',
          isDragActive
            ? 'border-signal-green bg-signal-green/5 scale-[1.01]'
            : 'border-ink-700 hover:border-ink-500 bg-ink-900/50'
        )}
      >
        <input {...getInputProps()} />
        <div className="text-3xl mb-3">{isDragActive ? '◎' : '◈'}</div>
        <p className="text-ink-300 text-sm mb-1">
          {isDragActive ? 'Drop to ingest' : 'Drop documents here'}
        </p>
        <p className="text-ink-500 text-xs">
          .txt · .md · .pdf · .json — up to 20 files
        </p>
      </div>

      {uploading && (
        <div className="flex items-center gap-2 text-xs text-ink-400 px-1">
          <span className="animate-spin">◌</span>
          <span>Chunking & embedding...</span>
        </div>
      )}

      {result && (
        <div className="px-3 py-2 rounded bg-signal-green/10 border border-signal-green/20 text-xs text-signal-green">
          ✓ {result.files_processed} file(s) · {result.chunks_ingested} chunks ingested · {result.total_in_store} total in store
        </div>
      )}

      {error && (
        <div className="px-3 py-2 rounded bg-signal-red/10 border border-signal-red/20 text-xs text-signal-red">
          ⚠ {error}
        </div>
      )}

      {acceptedFiles.length > 0 && !uploading && (
        <div className="space-y-1">
          {acceptedFiles.map(f => (
            <div key={f.name} className="flex items-center gap-2 text-xs text-ink-400">
              <span className="text-ink-600">◇</span>
              <span>{f.name}</span>
              <span className="text-ink-600">({(f.size / 1024).toFixed(1)} KB)</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

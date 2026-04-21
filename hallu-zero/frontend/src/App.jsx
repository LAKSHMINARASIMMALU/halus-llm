import React, { useEffect, useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import { Sidebar } from './components/Sidebar'
import { ChatPage } from './pages/ChatPage'
import { IngestPage } from './pages/IngestPage'
import { DashboardPage } from './pages/DashboardPage'
import { TrainingPage } from './pages/TrainingPage'
import { chatApi } from './lib/api'

export default function App() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    chatApi.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))

    // Poll health every 30s
    const id = setInterval(() => {
      chatApi.health()
        .then(setHealth)
        .catch(() => setHealth({ status: 'error' }))
    }, 30000)

    return () => clearInterval(id)
  }, [])

  return (
    <div className="flex min-h-screen bg-ink-950">
      {/* Subtle scanline overlay */}
      <div className="fixed inset-0 scanline pointer-events-none z-50 opacity-30" />

      <Sidebar health={health} />

      <main className="flex-1 min-w-0">
        <Routes>
          <Route path="/"          element={<ChatPage />} />
          <Route path="/ingest"    element={<IngestPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/training"  element={<TrainingPage />} />
        </Routes>
      </main>
    </div>
  )
}

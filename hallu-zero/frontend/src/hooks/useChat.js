import { useState, useCallback, useRef, useEffect } from 'react'
import { chatApi } from '../lib/api'

const STORAGE_KEY = 'hallu_zero_messages'

function loadMessages() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveMessages(messages) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-100)))
  } catch { localStorage.removeItem(STORAGE_KEY) }
}

export function useChat() {
  const [messages, setMessages] = useState(() => loadMessages())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('auto') // 'auto' | 'rag' | 'direct'
  const sessionId = useRef(crypto.randomUUID())

  useEffect(() => { saveMessages(messages) }, [messages])

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || loading) return
    const userMsg = {
      id: crypto.randomUUID(), role: 'user',
      content: query, timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setError(null)
    try {
      const result = await chatApi.send(query, sessionId.current, mode)
      const assistantMsg = {
        id: crypto.randomUUID(), role: 'assistant',
        content: result.response,
        sources: result.sources || [],
        verification: result.verification || {},
        confidence: result.confidence || 0,
        attempts: result.attempts || 1,
        model: result.model || '',
        mode: result.mode || 'direct',
        timestamp: new Date().toISOString(),
        sessionId: result.session_id,
        query,
      }
      setMessages(prev => [...prev, assistantMsg])
      return assistantMsg
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Request failed'
      setError(msg)
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(), role: 'error',
        content: msg, timestamp: new Date().toISOString(),
      }])
    } finally {
      setLoading(false)
    }
  }, [loading, mode])

  const submitFeedback = useCallback(async (message, rating, comment = '') => {
    try {
      await chatApi.feedback({
        session_id: message.sessionId || sessionId.current,
        query: message.query || '',
        response: message.content,
        rating, comment,
        confidence_score: message.confidence || 0,
        verification_passed: message.verification?.passed || false,
        context_sources: (message.sources || []).map(s => s.source),
      })
      setMessages(prev => prev.map(m =>
        m.id === message.id ? { ...m, feedbackGiven: rating } : m
      ))
    } catch (err) { console.error('Feedback error:', err) }
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    localStorage.removeItem(STORAGE_KEY)
    sessionId.current = crypto.randomUUID()
    setError(null)
  }, [])

  return { messages, loading, error, mode, setMode, sendMessage, submitFeedback, clearMessages }
}
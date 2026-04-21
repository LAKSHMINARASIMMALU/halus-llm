import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,
})

export const chatApi = {
  send: (query, sessionId, mode = 'auto') =>
    api.post('/chat', { query, session_id: sessionId, mode }).then(r => r.data),

  feedback: (payload) =>
    api.post('/feedback', payload).then(r => r.data),

  ingest: (files) => {
    const form = new FormData()
    files.forEach(f => form.append('files', f))
    return api.post('/ingest', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(r => r.data)
  },

  stats: () =>
    api.get('/stats').then(r => r.data),

  health: () =>
    api.get('/health').then(r => r.data),

  trainingPairs: () =>
    api.get('/training/pairs').then(r => r.data),

  clearKnowledge: () =>
    api.delete('/knowledge').then(r => r.data),
}

export default api
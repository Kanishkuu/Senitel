# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sentinel** - An Insider Threat Detection System with real-time log streaming, threat analysis, and ML-based threat detection using a Temporal Transformer model.

## Development Commands

### Backend (Express.js + WebSocket)
```bash
cd backend
npm install
npm start      # Production (includes inference service)
npm run dev    # Development with nodemon
```
Backend runs on http://localhost:5000

### Frontend (React + Vite + Tailwind)
```bash
cd frontend
npm install
npm run dev    # Development server with hot reload
npm run build  # Production build
```
Frontend runs on http://localhost:3000 with API proxy to backend

### Streaming Real CERT Data

```bash
cd backend
pip install pandas requests flask werkzeug  # dependencies
python cert_streamer.py                  # Stream all events chronologically
python cert_streamer.py --sample 1000   # Sample 1000 events per file
```

## Architecture

### Data Flow
```
CERT CSV Files → cert_streamer.py → /api/stream/event → Backend
                                                          ↓
                                                   Transformer Model
                                                   (inference_service.py)
                                                          ↓
                                                   WebSocket Broadcast
                                                          ↓
                                                     Frontend React
```

### Key Files
- `backend/src/index.js` - Express server, starts inference service, broadcasts events
- `backend/inference_service.py` - Python Flask service for transformer model predictions
- `backend/cert_streamer.py` - Streams real CERT r4.2 events chronologically
- `frontend/src/App.jsx` - React dashboard with live log feed and alerts

### CERT Data Source (`data/r4.2/r4.2/`)
- `logon.csv` - User logon/logoff events
- `device.csv` - USB/removable media events
- `file.csv` - File access events
- `email.csv` - Email sent/received
- `http.csv` - Web browsing (large file)
- `psychometric.csv` - Big Five personality traits

### ML Model (`models/`)
- `transformer_model.pth` - Temporal Transformer for sequence-based threat detection
- The model maintains sliding window of events per user for context

### Threat Levels
- **normal**: 0-20% threat score
- **low**: 20-40%
- **medium**: 40-60% (visible in UI)
- **high**: 60-80% (visible in UI)
- **critical**: 80-100% (visible in UI)

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with inference status |
| POST | `/api/stream/event` | Receives events, returns model predictions |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/clients` | WebSocket connected clients count |

### Inference Service (Port 5001)
- Python Flask service
- Loads transformer model
- Maintains event sequences per user
- Returns threat score for each event

### Environment Variables
- Backend: `PORT` (default: 5000)
- Frontend: `VITE_WS_URL` (default: ws://localhost:5000)

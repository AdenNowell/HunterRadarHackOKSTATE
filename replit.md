# Hunter Radar - Compass-Based Real-time Hunter Tracking

## Overview
Hunter Radar is a tactical real-time hunter tracking application designed with a compass-based radar as its primary interface. It features a military-inspired camo design with high-visibility safety orange text for optimal outdoor readability. The radar dynamically rotates based on the device's orientation (compass heading), providing true bearing to other hunters. Key capabilities include real-time location updates, proximity alerts, and AI-powered safety features. The project aims to provide hunters with a robust, intuitive tool for maintaining situational awareness and enhancing safety in the field.

## User Preferences
None yet - future features may include:
- Preferred distance units (meters/feet)
- Custom alert thresholds
- Map style preferences
- Safety zone radius configuration

## System Architecture
The application is built with a React + TypeScript frontend utilizing Wouter for routing, TanStack Query for data fetching, Tailwind CSS for styling, and Shadcn UI for components. The backend uses Express.js with WebSockets for real-time communication. AI functionalities are powered by Google Gemini 2.5 Flash via Replit AI Integrations, and weather data is sourced from the OpenWeatherMap API. Device orientation is leveraged through the DeviceOrientation API for compass functionality.

**UI/UX Decisions:**
- **Design Theme**: Military-inspired woodland camouflage backgrounds with safety orange (#FF6600) for all text and interactive elements.
- **Typography**: Rajdhani for headings, Roboto Mono for precise data, and Inter for body text, chosen for a tactical aesthetic and outdoor readability.
- **Visuals**: Deer in the woods background with semi-transparent overlays and backdrop blur effects. High-contrast orange-on-camo for glove-friendly operation.
- **Tabbed Interface**: All features organized in tabs for easy access - Radar, Hunters, Weather, Game Board, and AI Coach tabs with icons and safety orange active state.
- **Radar Display**: Full-screen, compass-based radar rotating with device heading. Displays distance rings (0.33mi, 0.67mi, 1.0mi) and cardinal directions (N, E, S, W, with North highlighted in orange).
- **Hunter Visualization**: Color-coded dots for hunters based on distance (0-0.33mi red, 0.33-0.67mi amber, 0.67-1.0mi green).

**Technical Implementations:**
- **Real-time Tracking**: WebSocket-based live position updates for hunters.
- **Offline Functionality**: `localStorage` caching for positions and data, with an offline queue for updates and automatic synchronization when connectivity is restored.
- **Distance Calculations**: Haversine formula for accurate great-circle distances and forward azimuth for bearing.
- **AI Features**:
    - **Reg Coach**: FastAPI + Gemini RAG system for Oklahoma hunting regulations Q&A with citations. Backend runs on port 8000 (`regcoach_py/main.py`) using google-generativeai 0.8.5, text-embedding-004 for embeddings, and gemini-1.5-flash for generation. Frontend component in `client/src/components/RegCoach.tsx` calls `/ask` endpoint, displays answers with citations including title, jurisdiction, version, last reviewed date, and similarity scores.
    - **Team Briefing**: Morning safety and movement outlook incorporating weather data and previous day's spacing hotspots.
    - **Anomaly Explainer**: Detects stationary hunters, analyzes causes vs. risks, and provides safety recommendations.
- **Voice Alerts (TTS)**: ElevenLabs Text-to-Speech integration for spoken safety alerts. Provides voice notifications for proximity warnings, animal sightings, and SOS emergencies. Features include:
    - **Priority Queue**: SOS > urgent > normal playback order with interrupt logic
    - **Caching**: Generated MP3s cached in `public/tts/` to minimize API calls and improve response time
    - **Rate Limiting**: 5-second cooldown per event_id prevents duplicate alerts
    - **User Controls**: Enable/disable toggle, volume slider (0-100%), auto-play urgent alerts setting
    - **API Endpoints**: `/api/tts/event` for generation, `/api/tts/prewarm` for pre-caching common phrases
    - **Frontend Manager**: `client/src/lib/audioManager.ts` handles prioritized playback with autoplay fallback
- **Weather Widget**: Persistent display of current conditions and an 8-hour forecast, auto-refreshing every 5 minutes.
- **Game Board**: Feature for hunters to post harvest details (photos, species, weight, description, GPS location).

**System Design Choices:**
- **Modular Structure**: Clear separation of concerns into `client`, `server`, and `shared` directories.
- **Data Model**: Defined schemas for Hunter, Boundary, and GamePost entities.
- **WebSocket Protocol**: Specific message types for `position_update`, `hunter_joined`, `hunter_left`, and `boundary_update`.
- **GIS Integration (Future)**: Planning for public land boundary integration from BLM and USFS via ArcGIS REST services.

## External Dependencies
- **AI Integration**: Google Gemini 2.5 Flash (via Replit AI Integrations)
- **RAG Backend**: FastAPI (0.115.0) + Google Generative AI (0.8.5) for regulation Q&A
- **Text-to-Speech**: ElevenLabs API for voice alert generation
- **Weather Data**: OpenWeatherMap API
- **Mapping (Client-side)**: Google Maps JavaScript API (for potential future use, currently client-side only)
- **Storage**: In-memory storage (MemStorage)

## Running the RAG API Backend
The Python FastAPI backend provides the Reg Coach functionality. To start it manually:

```bash
# Install dependencies (if needed)
pip install -r regcoach_py/requirements.txt

# Start the API on port 8000
uvicorn regcoach_py.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Ingest docs: POST http://localhost:8000/ingest
- Ask questions: POST http://localhost:8000/ask
- TTS generation: POST http://localhost:8000/api/tts/event
- TTS prewarm: POST http://localhost:8000/api/tts/prewarm

The frontend automatically connects to the API on port 8000 when running locally.

## Environment Secrets

The following secrets must be configured in Replit Secrets:
- `GEMINI_API_KEY` - Required for RAG Q&A and embeddings
- `ELEVENLABS_API_KEY` - Required for voice alert generation
- `OPENWEATHER_API_KEY` - Required for weather data
- `GOOGLE_MAPS_API_KEY` - Required for mapping features
- `SESSION_SECRET` - Required for session management

Optional TTS configuration:
- `ELEVENLABS_VOICE_ID` - Voice for normal/urgent alerts (defaults to "21m00Tcm4TlvDq8ikWAM")
- `ELEVENLABS_EMERGENCY_VOICE_ID` - Voice for SOS alerts (defaults to same as VOICE_ID)
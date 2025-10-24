# Accelerated Implementation Plan: FastAPI Backend + React Frontend

This document outlines a focused plan for quickly replacing the current Streamlit application with a FastAPI backend and React frontend.

## Overview

The goal is to implement a production-ready application as quickly as possible, focusing only on the essential components:
- FastAPI backend with core endpoints
- React frontend with chat interface
- Integration between frontend and backend

## Timeline

This accelerated implementation will be completed in **6 weeks**.

## 1. Backend Development (Weeks 1-3)

### Week 1: FastAPI Foundation

#### Core Setup
- Initialize FastAPI project with proper structure
- Set up dependency management (requirements.txt, virtual env)
- Configure CORS for frontend communication
- Implement basic error handling and middleware

#### Authentication (Basic)
- Implement simple API key authentication
- Set up basic authorization middleware
- Create user management endpoints (if needed)

#### Initial Endpoints
- Health check and status endpoints
- Basic researcher information endpoints
- Simple query endpoint without streaming

```python
# Example FastAPI structure
app = FastAPI(title="Moffitt RAG API", version="1.0")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/researchers")
async def get_researchers(limit: int = 10, offset: int = 0):
    # Implementation...
    return {"researchers": [...]}

@app.post("/api/query")
async def process_query(query: QueryRequest):
    # Implementation...
    return {"result": "..."}
```

### Week 2: Core RAG Components

#### Vector Database Integration
- Refactor vector database service from Streamlit app
- Implement optimized search functionality
- Create utility functions for vector operations

#### Agent Implementation
- Port agent implementation from Streamlit app
- Refactor tool implementations for API context
- Implement query processing service

#### LLM Integration
- Port LLM provider code from Streamlit app
- Implement provider fallback strategies
- Set up environment-based configuration

### Week 3: Advanced API Features

#### Streaming Responses
- Implement WebSocket endpoint for streaming responses
- Create StreamingResponse endpoints for SSE alternative
- Add connection management for WebSockets

```python
@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Process query...
        # Stream results back
        for chunk in result_chunks:
            await websocket.send_text(chunk)
```

#### Middleware & Error Handling
- Implement comprehensive error handling
- Add request logging middleware
- Create rate limiting middleware

#### API Documentation
- Configure Swagger/OpenAPI documentation
- Add detailed endpoint descriptions
- Create examples for all endpoints

## 2. Frontend Development (Weeks 3-5)

### Week 3: React Project Setup

#### Project Initialization
- Create React project with TypeScript
- Set up project structure (components, hooks, services)
- Configure build system and linting

#### Core Components
- Create layout components (header, sidebar, main content)
- Implement basic UI theme and styling
- Set up routing and navigation

#### State Management
- Configure state management (React Context or Redux)
- Set up API service layer
- Implement authentication state and API key management

### Week 4: Chat Interface Development

#### Message Components
- Create message list and message item components
- Implement user and assistant message styling
- Add support for markdown rendering

```jsx
// Example Message component
function Message({ type, content }) {
  return (
    <div className={`message ${type}`}>
      <div className="avatar">
        {type === 'user' ? <UserIcon /> : <AssistantIcon />}
      </div>
      <div className="content">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    </div>
  );
}
```

#### Input Components
- Create chat input with submission handling
- Implement suggestions and autocomplete
- Add typing indicators and loading states

#### WebSocket Integration
- Implement WebSocket connection management
- Create message streaming visualization
- Add reconnection logic and error handling

### Week 5: Additional UI Features

#### Search & Filters
- Create researcher search interface
- Implement department/program filters
- Add basic pagination and result display

#### User Interface Refinement
- Improve responsive design
- Enhance accessibility features
- Add animations and transitions

#### Error Handling
- Implement error boundaries
- Create error message components
- Add retry functionality for failed operations

## 3. Integration and Deployment (Week 6)

### Week 6: Integration & Testing

#### End-to-End Integration
- Connect frontend to all backend endpoints
- Test WebSocket streaming
- Verify authentication flow

#### Deployment Setup
- Create Docker files for frontend and backend
- Set up Docker Compose for local deployment
- Configure production build process

#### Documentation
- Create API usage documentation
- Add setup and deployment instructions
- Document known issues and limitations

## Essential Components

### Backend Components

#### Core FastAPI App
```
backend/
  ├── app/
  │   ├── api/
  │   │   ├── endpoints/
  │   │   │   ├── researchers.py
  │   │   │   ├── query.py
  │   │   │   └── admin.py
  │   │   └── dependencies.py
  │   ├── core/
  │   │   ├── config.py
  │   │   └── security.py
  │   ├── db/
  │   │   ├── vector_store.py
  │   │   └── hybrid_search.py
  │   ├── models/
  │   │   ├── researcher.py
  │   │   └── query.py
  │   └── services/
  │       ├── agent.py
  │       ├── llm.py
  │       └── vector_search.py
  ├── main.py
  └── requirements.txt
```

#### Essential API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/researchers` | GET | List researchers |
| `/api/departments` | GET | List departments |
| `/api/programs` | GET | List programs |
| `/api/query` | POST | Process a query |
| `/ws/query` | WebSocket | Streaming query responses |
| `/api/admin/rebuild` | POST | Rebuild vector database |

### Frontend Components

#### Core React App
```
frontend/
  ├── public/
  ├── src/
  │   ├── components/
  │   │   ├── Layout/
  │   │   ├── Chat/
  │   │   │   ├── MessageList.tsx
  │   │   │   ├── MessageItem.tsx
  │   │   │   ├── ChatInput.tsx
  │   │   │   └── StreamingResponse.tsx
  │   │   └── Researchers/
  │   ├── hooks/
  │   │   ├── useWebSocket.ts
  │   │   ├── useQuery.ts
  │   │   └── useAuth.ts
  │   ├── services/
  │   │   ├── api.ts
  │   │   └── websocket.ts
  │   ├── store/
  │   │   ├── context.tsx
  │   │   ├── types.ts
  │   │   └── actions.ts
  │   ├── utils/
  │   │   ├── markdown.ts
  │   │   └── formatting.ts
  │   ├── App.tsx
  │   └── index.tsx
  ├── package.json
  └── tsconfig.json
```

#### Essential UI Components

| Component | Description |
|-----------|-------------|
| `Layout` | Main application layout |
| `MessageList` | Displays chat messages |
| `MessageItem` | Individual message rendering |
| `ChatInput` | User input for queries |
| `StreamingResponse` | Handles streaming responses |
| `SearchFilters` | Department/program filters |
| `ResearcherCard` | Displays researcher information |

## Minimal Deployment Configuration

### Docker Setup

#### Backend Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Dockerfile
```dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### Docker Compose
```yaml
version: '3'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - VECTOR_DB_PATH=/app/data/vector_db
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
```

## Development Best Practices

### Commit Strategy
- **Commit frequently** with focused changes
- Keep commits small and targeted to specific functionality
- Use descriptive commit messages that explain the purpose of the changes
- Aim for commits that can be easily reviewed and understood
- Create meaningful checkpoints that allow for easier debugging and rollback if needed

### Code Organization
- Follow consistent naming conventions
- Add inline comments for complex logic
- Include docstrings for all modules, classes, and functions
- Use meaningful variable and function names
- Maintain separation of concerns between layers

## Next Steps After Implementation

After completing this accelerated implementation, consider these next steps:

1. **Performance Optimization**
   - Implement caching for common queries
   - Optimize vector search algorithms
   - Add database indexing improvements

2. **Advanced Features**
   - User authentication system
   - Query history and saved searches
   - Advanced filtering and visualization

3. **Production Enhancements**
   - Comprehensive monitoring
   - Horizontal scaling capabilities
   - Advanced security features
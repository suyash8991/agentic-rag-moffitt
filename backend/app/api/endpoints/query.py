"""
Query API endpoints.

This module provides endpoints for executing queries against the RAG system.
"""

import asyncio
import uuid
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ...models.query import QueryRequest, QueryResponse, QueryStatus, StreamingMessage
from ...core.security import get_api_key
from ...services.agent import process_query, query_status

router = APIRouter()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


@router.post("/query", response_model=QueryResponse)
async def create_query(
    query_request: QueryRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Execute a query against the RAG system.

    This endpoint processes a query synchronously and returns the result.
    For streaming responses, use the WebSocket endpoint.

    Args:
        query_request: Query request parameters
        api_key: API key for authentication

    Returns:
        QueryResponse: Complete query response
    """
    if query_request.streaming:
        raise HTTPException(
            status_code=400,
            detail="Streaming is enabled. Use WebSocket endpoint for streaming responses.",
        )

    try:
        # Generate a unique ID for this query
        query_id = str(uuid.uuid4())

        # Process the query
        result = await process_query(
            query_id=query_id,
            query=query_request.query,
            query_type=query_request.query_type,
            streaming=False,
            max_results=query_request.max_results,
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/{query_id}", response_model=QueryStatus)
async def get_query_status(
    query_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    Get the status of a query.

    Args:
        query_id: ID of the query
        api_key: API key for authentication

    Returns:
        QueryStatus: Status of the query
    """
    try:
        status = query_status(query_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/{query_id}/stream")
async def stream_query_response(
    query_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    Stream a query response using server-sent events (SSE).

    This is an alternative to WebSockets for clients that don't support WebSockets.

    Args:
        query_id: ID of the query
        api_key: API key for authentication

    Returns:
        EventSourceResponse: SSE response stream
    """
    async def event_generator():
        try:
            # Check if the query exists
            status = query_status(query_id)
            if status is None:
                yield {
                    "event": "error",
                    "data": {"message": f"Query {query_id} not found"}
                }
                return

            # If the query is already completed, return the result immediately
            if status.completed:
                # TODO: Get the complete result and return it
                yield {
                    "event": "complete",
                    "data": {"message": "Query already completed", "query_id": query_id}
                }
                return

            # Otherwise, stream the results as they come in
            # This is a placeholder for actual implementation
            for i in range(5):  # Simulate streaming
                await asyncio.sleep(1)  # Simulate processing time
                yield {
                    "event": "update",
                    "data": {"message": f"Processing step {i+1}", "progress": (i+1)/5}
                }

            yield {
                "event": "complete",
                "data": {"message": "Query processing completed", "query_id": query_id}
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": {"message": str(e)}
            }

    return EventSourceResponse(event_generator())


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming query responses.

    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()

    # Generate a unique ID for this connection
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket

    try:
        # Process messages
        while True:
            # Wait for a message
            data = await websocket.receive_json()

            # Extract query information
            query = data.get("query")
            query_type = data.get("query_type", "general")
            max_results = data.get("max_results", 5)

            # Validate the query
            if not query:
                await websocket.send_json(
                    StreamingMessage(
                        message_type="error",
                        content="No query provided",
                        data={"error": "query_missing"}
                    ).model_dump()
                )
                continue

            # Generate a unique ID for this query
            query_id = str(uuid.uuid4())

            # Send acknowledgement
            await websocket.send_json(
                StreamingMessage(
                    message_type="info",
                    content="Query received",
                    data={"query_id": query_id}
                ).model_dump()
            )

            # This is a placeholder for the actual streaming implementation
            # In a real implementation, you would call process_query with callbacks
            # that send updates to the WebSocket

            # Simulate processing steps
            await websocket.send_json(
                StreamingMessage(
                    message_type="thought",
                    content="Analyzing the query to determine the best approach."
                ).model_dump()
            )
            await asyncio.sleep(1)

            await websocket.send_json(
                StreamingMessage(
                    message_type="tool_call",
                    content="Searching for relevant researchers",
                    data={"tool": "researcher_search", "input": {"topic": query}}
                ).model_dump()
            )
            await asyncio.sleep(2)

            await websocket.send_json(
                StreamingMessage(
                    message_type="tool_result",
                    content="Found 3 researchers matching the query",
                    data={"researchers": ["John Doe", "Jane Smith", "Bob Johnson"]}
                ).model_dump()
            )
            await asyncio.sleep(1)

            await websocket.send_json(
                StreamingMessage(
                    message_type="answer",
                    content=f"Based on my search, there are several researchers working on {query}. The most relevant are John Doe, Jane Smith, and Bob Johnson."
                ).model_dump()
            )

            await websocket.send_json(
                StreamingMessage(
                    message_type="complete",
                    content="Query processing completed",
                    data={"query_id": query_id}
                ).model_dump()
            )

    except WebSocketDisconnect:
        # Clean up on disconnect
        if connection_id in active_connections:
            del active_connections[connection_id]
    except Exception as e:
        # Send error message
        try:
            await websocket.send_json(
                StreamingMessage(
                    message_type="error",
                    content=f"Error processing query: {str(e)}",
                    data={"error": str(e)}
                ).model_dump()
            )
        except:
            pass

        # Clean up
        if connection_id in active_connections:
            del active_connections[connection_id]
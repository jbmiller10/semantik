"""WebSocket manager with Redis Streams for distributed state synchronization."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Set

import redis.asyncio as redis
from fastapi import WebSocket

from shared.config import settings

logger = logging.getLogger(__name__)


class RedisStreamWebSocketManager:
    """WebSocket manager that uses Redis Streams for distributed state synchronization."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        self.redis: redis.Redis | None = None
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        self.consumer_group = f"webui-{uuid.uuid4().hex[:8]}"
        self.redis_url = settings.REDIS_URL
        
    async def startup(self) -> None:
        """Initialize Redis connection on application startup."""
        try:
            self.redis = await redis.from_url(
                self.redis_url,
                decode_responses=True,
                health_check_interval=30,
                socket_keepalive=True,
                retry_on_timeout=True,
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"WebSocket manager connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        # Cancel all consumer tasks
        for job_id, task in list(self.consumer_tasks.items()):
            logger.info(f"Cancelling consumer task for job {job_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        # Close all WebSocket connections
        for key, websockets in list(self.connections.items()):
            for websocket in list(websockets):
                try:
                    await websocket.close()
                except Exception:
                    pass
                    
        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logger.info("WebSocket manager Redis connection closed")
            
    async def connect(self, websocket: WebSocket, job_id: str, user_id: str) -> None:
        """Handle new WebSocket connection."""
        await websocket.accept()
        
        # Store connection
        key = f"{user_id}:{job_id}"
        if key not in self.connections:
            self.connections[key] = set()
        self.connections[key].add(websocket)
        
        logger.info(f"WebSocket connected: user={user_id}, job={job_id}")
        
        # Get current job state from database and send it
        try:
            from shared.database.factory import create_job_repository
            job_repo = create_job_repository()
            job = await job_repo.get_job(job_id)
            
            if job:
                # Send current state
                state_message = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "current_state",
                    "data": {
                        "status": job["status"],
                        "total_files": job.get("total_files", 0),
                        "processed_files": job.get("processed_files", 0),
                        "failed_files": job.get("failed_files", 0),
                        "current_file": job.get("current_file"),
                        "error": job.get("error"),
                    }
                }
                await websocket.send_json(state_message)
                logger.info(f"Sent current state to client for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to send current state for job {job_id}: {e}")
        
        # Start consumer task if not exists
        if job_id not in self.consumer_tasks:
            task = asyncio.create_task(self._consume_updates(job_id))
            self.consumer_tasks[job_id] = task
            logger.info(f"Started consumer task for job {job_id}")
            
        # Send message history
        await self._send_history(websocket, job_id)
        
    async def disconnect(self, websocket: WebSocket, job_id: str, user_id: str) -> None:
        """Handle WebSocket disconnection."""
        key = f"{user_id}:{job_id}"
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]
                
        logger.info(f"WebSocket disconnected: user={user_id}, job={job_id}")
        
        # Stop consumer if no more connections for this job
        if not any(job_id in k for k in self.connections):
            if job_id in self.consumer_tasks:
                logger.info(f"Stopping consumer task for job {job_id} (no more connections)")
                self.consumer_tasks[job_id].cancel()
                try:
                    await self.consumer_tasks[job_id]
                except asyncio.CancelledError:
                    pass
                del self.consumer_tasks[job_id]
                
    async def send_job_update(self, job_id: str, update_type: str, data: dict) -> None:
        """Send an update to Redis Stream for a specific job.
        
        This method is called by Celery tasks to send updates.
        """
        if not self.redis:
            logger.warning("Redis not connected, cannot send job update")
            return
            
        stream_key = f"job:updates:{job_id}"
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": update_type,
            "data": data
        }
        
        try:
            # Add to stream with automatic ID
            await self.redis.xadd(
                stream_key,
                {"message": json.dumps(message)},
                maxlen=1000  # Keep last 1000 messages
            )
            
            # Set TTL on first message (24 hours)
            await self.redis.expire(stream_key, 86400)
            
            logger.debug(f"Sent update to stream {stream_key}: type={update_type}")
        except Exception as e:
            logger.error(f"Failed to send update to Redis stream: {e}")
            
    async def _consume_updates(self, job_id: str) -> None:
        """Consume updates from Redis Stream for a specific job."""
        stream_key = f"job:updates:{job_id}"
        
        try:
            # Create consumer group
            try:
                await self.redis.xgroup_create(
                    stream_key,
                    self.consumer_group,
                    id="0"
                )
                logger.info(f"Created consumer group {self.consumer_group} for stream {stream_key}")
            except Exception as e:
                # Group might already exist
                logger.debug(f"Consumer group might already exist: {e}")
                
            consumer_name = f"consumer-{job_id}"
            last_id = ">"  # Start reading new messages
            
            while True:
                try:
                    # Read from stream with blocking
                    messages = await self.redis.xreadgroup(
                        self.consumer_group,
                        consumer_name,
                        {stream_key: last_id},
                        count=10,
                        block=1000  # 1 second timeout
                    )
                    
                    if messages:
                        for stream, stream_messages in messages:
                            for msg_id, data in stream_messages:
                                try:
                                    # Parse message
                                    message = json.loads(data["message"])
                                    
                                    # Send to all connected clients for this job
                                    await self._broadcast_to_job(job_id, message)
                                    
                                    # Acknowledge message
                                    await self.redis.xack(stream_key, self.consumer_group, msg_id)
                                    
                                    logger.debug(f"Processed message {msg_id} for job {job_id}")
                                except Exception as e:
                                    logger.error(f"Error processing message {msg_id}: {e}")
                                    
                    await asyncio.sleep(0.1)  # Small delay between reads
                    
                except asyncio.CancelledError:
                    # Clean up consumer
                    try:
                        await self.redis.xgroup_delconsumer(
                            stream_key,
                            self.consumer_group,
                            consumer_name
                        )
                        logger.info(f"Cleaned up consumer {consumer_name}")
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    logger.error(f"Error in consumer loop for job {job_id}: {e}")
                    await asyncio.sleep(5)  # Wait before retry
                    
        except asyncio.CancelledError:
            logger.info(f"Consumer task cancelled for job {job_id}")
            raise
        except Exception as e:
            logger.error(f"Fatal error in consumer for job {job_id}: {e}")
            
    async def _send_history(self, websocket: WebSocket, job_id: str) -> None:
        """Send historical messages to newly connected client."""
        stream_key = f"job:updates:{job_id}"
        
        try:
            # Read last 100 messages
            messages = await self.redis.xrange(
                stream_key,
                min="-",
                max="+",
                count=100
            )
            
            for msg_id, data in messages:
                try:
                    message = json.loads(data["message"])
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send historical message: {e}")
                    
            if messages:
                logger.info(f"Sent {len(messages)} historical messages to client for job {job_id}")
                
        except Exception as e:
            logger.warning(f"Failed to send history for job {job_id}: {e}")
            
    async def _broadcast_to_job(self, job_id: str, message: dict) -> None:
        """Broadcast message to all connections for a job."""
        disconnected = []
        
        for key, websockets in list(self.connections.items()):
            if job_id in key:
                for websocket in list(websockets):
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send message to websocket: {e}")
                        disconnected.append((key, websocket))
                        
        # Clean up disconnected clients
        for key, websocket in disconnected:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]


# Global instance
ws_manager = RedisStreamWebSocketManager()
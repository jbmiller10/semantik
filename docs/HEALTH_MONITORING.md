# Health Monitoring

Project Semantik provides comprehensive health check endpoints for monitoring service status in production environments.

## Endpoints

### WebUI Service

#### Basic Health Check
- **Endpoint**: `GET /api/health/`
- **Description**: Simple health check to verify the service is running
- **Response**: 
  ```json
  {
    "status": "healthy"
  }
  ```

#### Embedding Service Health
- **Endpoint**: `GET /api/health/embedding`
- **Description**: Detailed health status of the embedding service
- **Response States**:
  - **Healthy**: Service is initialized and ready
    ```json
    {
      "status": "healthy",
      "initialized": true,
      "model": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "device": "cuda",
        "max_sequence_length": 512,
        "quantization": "float32"
      }
    }
    ```
  - **Unhealthy**: Service not initialized or errored
    ```json
    {
      "status": "unhealthy",
      "initialized": false,
      "message": "Embedding service not initialized"
    }
    ```
  - **Degraded**: Service initialized but experiencing issues
    ```json
    {
      "status": "degraded",
      "initialized": true,
      "error": "Failed to retrieve model information"
    }
    ```

#### Readiness Check
- **Endpoint**: `GET /api/health/ready`
- **Description**: Kubernetes-style readiness probe that verifies the service can handle requests
- **Response**:
  ```json
  {
    "ready": true,
    "status": "Service is ready to handle requests"
  }
  ```

### VecPipe Service

#### Comprehensive Health Check
- **Endpoint**: `GET /health`
- **Description**: Detailed health status of all VecPipe components
- **Response Example**:
  ```json
  {
    "status": "healthy",
    "components": {
      "qdrant": {
        "status": "healthy",
        "collections_count": 5
      },
      "embedding": {
        "status": "healthy",
        "model": "BAAI/bge-base-en-v1.5",
        "dimension": 768
      }
    }
  }
  ```

#### Status Values
- **healthy**: All components functioning normally
- **degraded**: Some non-critical components have issues but service is operational
- **unhealthy**: Critical components are down, service cannot function properly

## Monitoring Integration

### Kubernetes

Example readiness and liveness probes:

```yaml
livenessProbe:
  httpGet:
    path: /api/health/
    port: 5555
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/health/ready
    port: 5555
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Prometheus Metrics

The health endpoints can be scraped by Prometheus for monitoring:

```yaml
- job_name: 'semantik-webui'
  static_configs:
    - targets: ['webui:5555']
  metrics_path: '/api/health/embedding'
  
- job_name: 'semantik-vecpipe'  
  static_configs:
    - targets: ['vecpipe:8080']
  metrics_path: '/health'
```

### Docker Compose Health Checks

```yaml
services:
  webui:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5555/api/health/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
      
  vecpipe:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Alerting

Example alert rules for Prometheus:

```yaml
groups:
  - name: semantik
    rules:
      - alert: EmbeddingServiceDown
        expr: |
          probe_success{job="semantik-webui", instance="/api/health/embedding"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Embedding service is down"
          
      - alert: QdrantConnectionFailure
        expr: |
          probe_success{job="semantik-vecpipe", instance="/health"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "VecPipe cannot connect to Qdrant"
```

## Best Practices

1. **Monitor all endpoints**: Set up monitoring for both WebUI and VecPipe health endpoints
2. **Use appropriate timeouts**: Health checks should respond quickly (< 5 seconds)
3. **Check dependencies**: The comprehensive health checks verify external dependencies like Qdrant
4. **Separate liveness and readiness**: Use `/api/health/` for liveness and `/api/health/ready` for readiness
5. **Alert on degraded state**: Don't just monitor for complete failures, alert on degraded states too
<docker-configuration>
  <purpose>Docker containerization and orchestration</purpose>
  <files>
    - docker-compose.yml: Base configuration
    - docker-compose.dev.yml: Development overrides
    - docker-compose.prod.yml: Production settings
    - docker-compose.cuda.yml: GPU support
    - Dockerfile: Multi-stage build
  </files>
</docker-configuration>

<services>
  <service name="webui">
    <image>semantik-webui</image>
    <ports>8001:8001</ports>
    <depends-on>postgres, redis, qdrant</depends-on>
    <healthcheck>GET /health/live</healthcheck>
  </service>
  
  <service name="vecpipe">
    <image>semantik-vecpipe</image>
    <ports>8002:8002</ports>
    <gpu>Optional CUDA support</gpu>
    <memory>4GB minimum</memory>
  </service>
  
  <service name="worker">
    <image>semantik-worker</image>
    <command>celery -A packages.webui.celery_app worker</command>
    <scale>Can run multiple instances</scale>
  </service>
  
  <service name="postgres">
    <image>postgres:16</image>
    <volume>postgres_data:/var/lib/postgresql/data</volume>
    <healthcheck>pg_isready</healthcheck>
  </service>
  
  <service name="redis">
    <image>redis:7-alpine</image>
    <volume>redis_data:/data</volume>
  </service>
  
  <service name="qdrant">
    <image>qdrant/qdrant:v1.12.5</image>
    <volume>qdrant_data:/qdrant/storage</volume>
    <ports>6333:6333</ports>
  </service>
</services>

<deployment-commands>
  <development>
    # Backend only (use local frontend)
    make docker-dev-up
    
    # Full stack
    make docker-up
  </development>
  
  <production>
    docker compose --profile prod up -d
  </production>
  
  <gpu-enabled>
    docker compose -f docker-compose.yml -f docker-compose.cuda.yml up
  </gpu-enabled>
</deployment-commands>

<environment-variables>
  <required>
    - JWT_SECRET: Authentication secret
    - DATABASE_URL: PostgreSQL connection
    - REDIS_URL: Redis connection
    - QDRANT_URL: Vector DB endpoint
  </required>
  
  <optional>
    - CUDA_VISIBLE_DEVICES: GPU selection
    - MODEL_CACHE_DIR: Model storage path
    - LOG_LEVEL: debug/info/warning/error
  </optional>
</environment-variables>

<volumes>
  <persistent>
    - postgres_data: Database storage
    - redis_data: Cache persistence  
    - qdrant_data: Vector storage
    - models_cache: ML models
  </persistent>
  
  <bind-mounts>
    - ./data:/app/data: Document storage
    - ./logs:/app/logs: Application logs
  </bind-mounts>
</volumes>

<networking>
  <network>semantik_network</network>
  <internal-communication>Service names as hostnames</internal-communication>
  <security>Only expose necessary ports</security>
</networking>

<common-issues>
  <issue>
    <problem>Permission denied on volumes</problem>
    <solution>Run scripts/fix-permissions.sh</solution>
  </issue>
  <issue>
    <problem>GPU not detected</problem>
    <solution>Install nvidia-docker2, use cuda compose file</solution>
  </issue>
  <issue>
    <problem>Migrations not running</problem>
    <solution>Check docker-entrypoint.sh execution</solution>
  </issue>
</common-issues>
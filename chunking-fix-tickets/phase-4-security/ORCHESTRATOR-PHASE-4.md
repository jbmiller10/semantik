# ORCHESTRATOR PHASE 4: Security & Performance Hardening

## Phase Overview
**Priority**: HIGH - Security vulnerabilities must be fixed  
**Total Duration**: 10 hours  
**Risk Level**: HIGH - Active vulnerabilities in production  
**Success Gate**: Security scan shows zero high/critical issues

## Context
Multiple security vulnerabilities exist: ReDoS attacks possible, XSS via insufficient sanitization, missing access controls, and no resource limits. This phase hardens the application against attacks while improving performance.

## Execution Strategy

### Pre-Flight Checklist
- [ ] Security scanning tools ready (OWASP ZAP, Semgrep)
- [ ] Performance profiling tools configured
- [ ] Test payloads prepared (XSS, ReDoS, etc.)
- [ ] Resource monitoring enabled
- [ ] Security team notified

### Ticket Execution Order

#### Stage 1: ReDoS Prevention (3 hours)
**Ticket**: SEC-001 - Fix ReDoS Vulnerabilities

**Priority**: CRITICAL - Active DoS vector

**Critical Actions**:
1. Install RE2 for linear-time regex
2. Replace all vulnerable patterns
3. Add regex timeout protection
4. Implement input validation
5. Add performance monitoring

**Validation**:
```python
# Test ReDoS protection
def test_redos_prevention():
    # Known ReDoS payloads
    evil_inputs = [
        "a" * 10000 + "!" * 10000,  # Catastrophic backtracking
        "x" * 50000 + "y",  # Exponential time
        "(((" * 1000 + ")))" * 1000,  # Nested groups
    ]
    
    for evil_input in evil_inputs:
        start = time.time()
        try:
            result = safe_regex.match_with_timeout(
                pattern=r'(a+)+b',
                text=evil_input,
                timeout=1.0
            )
        except RegexTimeout:
            pass  # Expected
        
        duration = time.time() - start
        assert duration < 1.1, f"Regex took {duration}s, should timeout at 1s"
    
    print("✓ ReDoS protection validated")
```

**Security Scan**:
```bash
# Scan for regex vulnerabilities
semgrep --config=auto --severity=ERROR --json | jq '.results[] | select(.check_id | contains("regex"))'

# Should return empty
```

#### Stage 2: HTML Sanitization (2 hours)
**Ticket**: SEC-002 - Implement Proper HTML Sanitization

**Parallel Execution**: Can run parallel with SEC-003

**Critical Actions**:
1. Replace string replacement with bleach/DOMPurify
2. Define strict tag/attribute allowlists
3. Sanitize at display time
4. Add CSP headers
5. Validate markdown rendering

**Validation**:
```python
# XSS prevention test
def test_xss_prevention():
    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "javascript:alert('XSS')",
        "<iframe src=javascript:alert('XSS')>",
        "';alert('XSS');//",
    ]
    
    for payload in xss_payloads:
        sanitized = sanitize_content(payload)
        assert "<script" not in sanitized
        assert "javascript:" not in sanitized
        assert "onerror" not in sanitized
        assert "onload" not in sanitized
    
    print("✓ XSS prevention validated")
```

**Browser Test**:
```javascript
// Test XSS in browser console
const testXSS = () => {
  const payloads = [
    '<img src=x onerror="alert(1)">',
    '<script>alert(2)</script>',
    'javascript:alert(3)'
  ];
  
  payloads.forEach(payload => {
    // Try to inject
    document.querySelector('[data-testid="chunk-content"]').innerHTML = payload;
    
    // Check if sanitized
    const content = document.querySelector('[data-testid="chunk-content"]').innerHTML;
    console.assert(!content.includes('script'), 'Script tags blocked');
    console.assert(!content.includes('onerror'), 'Event handlers blocked');
  });
  
  console.log('✓ XSS protection validated in browser');
};
```

#### Stage 3: Access Control (2 hours)
**Ticket**: SEC-003 - Add Document Access Validation

**Parallel Execution**: Can run parallel with SEC-002

**Critical Actions**:
1. Implement permission checks before operations
2. Add row-level security
3. Create audit log for access attempts
4. Cache permissions (5 min TTL)
5. Validate collection ownership

**Validation**:
```python
# Access control test
async def test_access_control():
    # Create users with different permissions
    user1 = create_user("user1")
    user2 = create_user("user2")
    
    # User1 creates a document
    doc = await create_document(user1, "private.txt")
    
    # User2 tries to access
    with pytest.raises(PermissionDeniedException):
        await preview_chunks(user2, doc.id)
    
    # Check audit log
    audit_entries = await get_audit_log(doc.id)
    assert any(e.user_id == user2.id and e.result == "DENIED" 
              for e in audit_entries)
    
    print("✓ Access control validated")
```

**SQL Verification**:
```sql
-- Check row-level security
SELECT 
    c.collection_id,
    c.user_id,
    p.permission
FROM collections c
LEFT JOIN user_collection_permissions p 
    ON c.id = p.collection_id
WHERE p.user_id = 'test_user';

-- Should only show permitted collections
```

#### Stage 4: Resource Limits (3 hours)
**Ticket**: SEC-004 - Implement Resource Limits

**Critical Actions**:
1. Set memory limits (500MB per operation)
2. Configure CPU limits via cgroups
3. Add operation timeouts (5 min)
4. Limit concurrent operations (3 per user)
5. Implement queue limits

**Validation**:
```python
# Resource limit test
async def test_resource_limits():
    # Test memory limit
    with pytest.raises(MemoryLimitExceeded):
        await process_document(size=600*1024*1024)  # 600MB
    
    # Test concurrent operations limit
    operations = []
    for i in range(5):
        op = asyncio.create_task(start_chunking_operation())
        operations.append(op)
    
    results = await asyncio.gather(*operations, return_exceptions=True)
    
    # First 3 should succeed, last 2 should be rejected
    assert sum(1 for r in results if not isinstance(r, Exception)) == 3
    assert sum(1 for r in results if isinstance(r, ConcurrencyLimitExceeded)) == 2
    
    print("✓ Resource limits validated")
```

**System Monitoring**:
```bash
# Monitor resource usage during test
while true; do
    echo "=== $(date) ==="
    # Memory usage
    ps aux | grep chunking | awk '{sum+=$6} END {print "Memory:", sum/1024, "MB"}'
    
    # CPU usage
    top -bn1 | grep chunking | awk '{sum+=$9} END {print "CPU:", sum, "%"}'
    
    # Check cgroups
    cat /sys/fs/cgroup/memory/chunking/memory.usage_in_bytes
    
    sleep 5
done
```

### Security Testing Protocol

#### Penetration Testing Checklist
```bash
# 1. ReDoS Testing
python redos_fuzzer.py --target http://localhost:8000/api/v2/chunking

# 2. XSS Testing
nikto -h http://localhost:8000 -Plugins "xss"

# 3. SQL Injection
sqlmap -u "http://localhost:8000/api/v2/chunking/preview" --data='{"strategy":"test"}' --level=5

# 4. Authentication Bypass
python auth_bypass_tester.py --endpoints /api/v2/chunking/*

# 5. Resource Exhaustion
locust -f resource_exhaustion.py --users 100 --spawn-rate 10
```

#### OWASP Top 10 Validation
```python
# Run OWASP ZAP scan
def run_security_scan():
    zap = ZAPv2(proxies={'http': 'http://127.0.0.1:8080'})
    
    # Spider the application
    zap.spider.scan('http://localhost:8000')
    while int(zap.spider.status()) < 100:
        time.sleep(2)
    
    # Active scan
    zap.ascan.scan('http://localhost:8000')
    while int(zap.ascan.status()) < 100:
        time.sleep(5)
    
    # Get results
    alerts = zap.core.alerts()
    high_risk = [a for a in alerts if a['risk'] in ['High', 'Critical']]
    
    assert len(high_risk) == 0, f"Found {len(high_risk)} high risk vulnerabilities"
    
    print("✓ OWASP scan passed")
```

### Performance Impact Assessment

```python
# Measure performance impact of security fixes
async def measure_security_overhead():
    baseline = await measure_performance(security_enabled=False)
    secured = await measure_performance(security_enabled=True)
    
    overhead = {
        'latency': (secured['p99'] - baseline['p99']) / baseline['p99'] * 100,
        'throughput': (baseline['rps'] - secured['rps']) / baseline['rps'] * 100,
        'cpu': secured['cpu'] - baseline['cpu'],
        'memory': secured['memory'] - baseline['memory']
    }
    
    print(f"Security overhead: {overhead}")
    
    # Acceptable overhead
    assert overhead['latency'] < 10, "Latency overhead < 10%"
    assert overhead['throughput'] < 10, "Throughput impact < 10%"
    assert overhead['cpu'] < 20, "CPU overhead < 20%"
    assert overhead['memory'] < 100, "Memory overhead < 100MB"
```

### Compliance Validation

#### GDPR Compliance
```python
def validate_gdpr_compliance():
    # Data minimization
    assert not contains_unnecessary_pii(chunk_metadata)
    
    # Right to erasure
    user_id = "test_user"
    await delete_user_data(user_id)
    remaining = await find_user_data(user_id)
    assert len(remaining) == 0
    
    # Audit logging
    logs = await get_audit_logs(user_id)
    assert all(log.has_timestamp for log in logs)
    assert all(log.has_purpose for log in logs)
```

#### SOC 2 Requirements
```python
def validate_soc2_requirements():
    # Encryption at rest
    assert database_encrypted()
    assert backups_encrypted()
    
    # Encryption in transit
    assert all_endpoints_use_https()
    assert websocket_uses_wss()
    
    # Access controls
    assert role_based_access_enabled()
    assert mfa_available()
    
    # Monitoring
    assert security_events_logged()
    assert anomaly_detection_enabled()
```

### Rollback Procedures

#### If Security Fixes Break Functionality
```python
# Feature flag for each security control
SECURITY_FLAGS = {
    'regex_timeout': True,
    'input_sanitization': True,
    'access_control': True,
    'resource_limits': True
}

def selective_rollback(feature):
    SECURITY_FLAGS[feature] = False
    logger.warning(f"Security feature {feature} disabled due to issues")
    
    # Compensating controls
    if feature == 'regex_timeout':
        enable_rate_limiting(aggressive=True)
    elif feature == 'input_sanitization':
        enable_waf_rules()
```

### Success Criteria

#### Security Requirements
- [ ] Zero ReDoS vulnerabilities
- [ ] Zero XSS vulnerabilities  
- [ ] All operations check permissions
- [ ] Resource limits enforced
- [ ] Security scan clean

#### Performance Requirements
- [ ] Regex operations < 100ms
- [ ] Sanitization overhead < 10ms
- [ ] Permission checks < 50ms
- [ ] No performance regression > 10%

#### Compliance
- [ ] OWASP Top 10 addressed
- [ ] GDPR compliant
- [ ] SOC 2 controls in place
- [ ] Security headers configured
- [ ] Audit logging complete

### Post-Phase Security Audit

```bash
#!/bin/bash
# Comprehensive security audit

echo "=== Security Audit Phase 4 ==="

# 1. Dependency scanning
echo "Checking dependencies..."
safety check
npm audit

# 2. Static analysis
echo "Running static analysis..."
bandit -r packages/
semgrep --config=auto

# 3. Dynamic scanning  
echo "Running dynamic scan..."
python owasp_zap_scan.py

# 4. Configuration audit
echo "Checking security configuration..."
python security_config_audit.py

# 5. Generate report
python generate_security_report.py > security_audit_phase4.pdf

echo "=== Audit Complete ==="
```

### Handoff to Phase 5

#### Security Deliverables
1. All vulnerabilities remediated
2. Security controls implemented
3. Audit trail enabled
4. Resource limits enforced
5. Compliance documented

#### Security Documentation
```markdown
## Security Measures Implemented

### Input Validation
- RE2 regex engine for DoS prevention
- Bleach HTML sanitization
- Input size limits
- Character encoding validation

### Access Control
- Row-level security
- Permission caching
- Audit logging
- Session management

### Resource Protection
- Memory limits: 500MB/operation
- CPU limits: 2 cores
- Time limits: 5 minutes
- Concurrency limits: 3/user
```

## Notes for Orchestrating Agent

**Security Principles**:
1. Security is NOT optional
2. Performance can be sacrificed for security
3. Log everything suspicious
4. Fail closed, not open
5. Defense in depth

**Testing Requirements**:
- Manual penetration testing
- Automated security scanning
- Fuzzing all inputs
- Load testing with limits
- Compliance validation

**Red Flags**:
- Any regex taking > 100ms
- Unsanitized user content
- Missing permission checks
- Unlimited resource usage
- No audit logging

This phase is critical for production readiness. A single vulnerability could compromise the entire system. Test thoroughly and err on the side of caution.
# Orchestrator Agent Instructions - Chunking Feature Implementation

You are an orchestrator agent responsible for managing the implementation of Phase 1 of the chunking feature fix. Your role is to coordinate multiple subagents to complete all tickets in your phase while ensuring quality and adherence to standards.

## Your Primary Responsibilities

1. **Phase Management**: Read all documentation, coordinate subagents, ensure quality
2. **Git Workflow**: Create branch, manage commits, create PR
3. **Quality Assurance**: Enforce reviews, testing, and linting
4. **Communication**: Report progress and blockers

---

## CRITICAL CONTEXT

**This is a PRE-RELEASE application**. We have complete freedom to:
- DELETE old code (don't wrap or maintain compatibility)
- Drop and recreate database tables
- Change any interface
- Implement the CORRECT solution from scratch

Key instruction for all subagents: **When you see old code, DELETE IT and replace it with the right implementation.**

---

## Step-by-Step Execution Process

### Phase 1: Initial Setup (First 30 minutes)

1. **Read Master Documentation**:
   ```
   /chunking-implementation/ORCHESTRATOR_MASTER_GUIDE.md
   /chunking-implementation/README.md
   ```

2. **Read Your Phase-Specific Documentation**:
   ```
   /chunking-implementation/phase-[X]-[name]/ORCHESTRATOR-PHASE-[X].md
   /chunking-implementation/phase-[X]-[name]/TICKET-*.md (all tickets)
   ```

3. **Create Phase Branch**:
   ```bash
   git checkout feature/improve-chunking
   git pull origin feature/improve-chunking
   git checkout -b phase-[X]-[description]-[date]
   # Example: git checkout -b phase-0-security-fixes-2025-01-09
   ```

### Phase 2: Task Assignment & Execution

For each ticket in your phase:

1. **Assign to Implementation Subagent**:
   ```
   "Please implement TICKET-[ID] from /chunking-implementation/phase-[X]-[name]/TICKET-[ID].md
   
   Critical reminders:
   - We are PRE-RELEASE - delete old code, don't wrap it
   - Follow the acceptance criteria exactly
   - Run tests for your changes
   - Ensure code is properly linted
   
   Key technical decisions:
   - 100 direct LIST partitions (NO virtual mapping)
   - AsyncIO streaming with 64KB buffers
   - Redis Pub/Sub for WebSockets
   - Modular monolith architecture
   
   Report back with:
   1. Files modified/created/deleted
   2. Test results
   3. Any concerns or blockers"
   ```

2. **After Implementation, Call Review Subagent**:
   ```
   "Please review the implementation of TICKET-[ID].
   
   Review checklist:
   - [ ] Acceptance criteria from ticket are met
   - [ ] No backwards compatibility code (we're pre-release)
   - [ ] Security best practices followed
   - [ ] Performance within specified bounds
   - [ ] Tests cover the changes
   - [ ] Code follows project patterns
   - [ ] No over-engineering
   
   For Phase 2 (Streaming) specifically check:
   - [ ] UTF-8 boundaries handled correctly
   - [ ] Memory stays under 100MB limit
   
   Report:
   1. APPROVED or NEEDS CHANGES
   2. If changes needed, list specific valid concerns
   3. Suggestions for improvement"
   ```

3. **Address Valid Review Concerns**:
   - If review finds VALID issues, assign back to implementation agent
   - Ignore over-engineering suggestions or backwards compatibility requests
   - Focus on security, correctness, and performance

### Phase 3: Quality Assurance

Before moving to next ticket:

1. **Run Tests** (delegate to subagent):
   ```
   "Run all relevant tests and ensure they pass:
   - pytest tests/[relevant_module]/ -v
   - npm run test (if frontend changes)
   Report any failures"
   ```

2. **Run Linting** (delegate to subagent):
   ```
   "Run linting and fix any issues:
   - Python: poetry run ruff check --fix packages/
   - Python: poetry run black packages/
   - TypeScript: npm run lint:fix
   Report any unfixable issues"
   ```

### Phase 4: Phase Completion

After all tickets are complete:

1. **Final Validation** (delegate to subagent):
   ```
   "Perform final validation for Phase [X]:
   
   Phase-specific checks:
   [Include relevant checks from ORCHESTRATOR-PHASE-[X].md success criteria]
   
   General checks:
   - All tests passing: pytest tests/ -v
   - Linting clean: poetry run ruff check packages/
   - No security vulnerabilities: npm run security:scan
   - Documentation updated if needed
   
   Report results"
   ```

2. **Commit Changes**:
   ```bash
   # Stage all changes
   git add -A
   
   # Create comprehensive commit message
   git commit -m "Phase [X]: [Description]
   
   Completed tickets:
   - TICKET-[ID1]: [Brief description]
   - TICKET-[ID2]: [Brief description]
   
   Key changes:
   - [Major change 1]
   - [Major change 2]
   
   All tests passing, security scan clean, linting complete.
   
   Co-Authored-By: [Agent IDs who worked on this]"
   ```

3. **Create Pull Request**:
   ```bash
   # Push branch
   git push origin phase-[X]-[description]-[date]
   
   # Create PR
   gh pr create \
     --base feature/improve-chunking \
     --title "Phase [X]: [Description]" \
     --body "## Summary
   Implements Phase [X] of chunking feature fix.
   
   ## Completed Tickets
   - [x] TICKET-[ID1]: [Description]
   - [x] TICKET-[ID2]: [Description]
   
   ## Key Changes
   - [Detail major changes]
   
   ## Testing
   - [x] All tests passing
   - [x] Security scan clean
   - [x] Linting complete
   - [x] Phase success criteria met
   
   ## Phase Success Criteria
   [Copy from ORCHESTRATOR-PHASE-[X].md]
   
   ## Review Checklist
   - [ ] Code follows project patterns
   - [ ] No backwards compatibility code
   - [ ] Performance within bounds
   - [ ] Security best practices followed"
   ```

---

## Parallel Execution Guidelines

If your phase has tickets that can run in parallel:

1. **Launch Multiple Subagents Simultaneously**:
   - Assign different tickets to different implementation agents
   - Ensure agents know about potential conflicts
   - Have agents coordinate on shared interfaces

2. **Coordinate Integration**:
   - When parallel work completes, test integration
   - Resolve any conflicts before proceeding

Example for Phase 0:
```
Morning: Launch SEC-001 (Agent A) and SEC-002 (Agent B) in parallel
Afternoon: After both complete, launch SEC-003
```

---

## Communication Templates

### Phase Start Message:
```
Starting Phase [X]: [Name]
Branch created: phase-[X]-[description]-[date]
Tickets to complete: [List]
Parallel execution plan: [Details]
Estimated completion: [Time]
```

### Progress Update:
```
Phase [X] Progress Update:
✅ Completed: TICKET-[ID] - [Description]
🔄 In Progress: TICKET-[ID] - [Description]
📋 Remaining: [Count] tickets
Blockers: [Any blockers]
On track for completion: [Yes/No]
```

### Phase Completion Message:
```
Phase [X] Complete: [Name]
✅ All [N] tickets implemented and reviewed
✅ Tests passing: [Count] tests
✅ Security scan: Clean
✅ Linting: No issues
✅ PR created: #[PR number]
Success criteria met: [Details]
Ready for next phase
```

---

## Important Reminders

1. **Pre-Release Advantage**: Constantly remind subagents we can DELETE old code
2. **Critical Technical Decisions**:
   - 100 direct partitions (NO virtual mapping)
   - AsyncIO streaming (handle UTF-8 boundaries!)
   - Simple solutions over complex
3. **Quality Gates**: Don't proceed unless tests pass and reviews are addressed
4. **Documentation**: Update docs if implementation differs from plan

---

## Escalation Triggers

Escalate immediately if:
- Security vulnerability not fully fixed
- UTF-8 corruption in streaming implementation
- Tests failing after multiple attempts
- Subagent blocked for >2 hours
- Subagent trying to maintain backwards compatibility

---

## Your Success Metrics

Your phase is successful when:
1. All tickets implemented and reviewed
2. All tests passing
3. Security scan clean (especially Phase 0)
4. Linting complete with no errors
5. PR created and ready for merge
6. Phase-specific success criteria met (see your ORCHESTRATOR-PHASE-[X].md)

---

**Remember: We are PRE-RELEASE. This is our opportunity to build it RIGHT. Use this freedom to create clean, secure, scalable solutions without legacy baggage.**

Good luck, Orchestrator!

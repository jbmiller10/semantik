# MCP Integration Testing Checklist

Manual testing checklist for verifying MCP integration functionality.

## Setup

- [ ] Fresh Semantik instance is running
- [ ] User account created and logged in
- [ ] At least one collection created with indexed documents
- [ ] Claude Desktop or other MCP client installed

## Profile Management

### Create Profile
- [ ] Create new profile with valid name (lowercase, alphanumeric, hyphens, underscores)
- [ ] Verify profile name validation rejects invalid characters
- [ ] Add description (verify character limit works)
- [ ] Select one or more collections
- [ ] Verify validation prevents empty collection selection
- [ ] Set search type and verify description updates
- [ ] Toggle reranker setting
- [ ] Configure advanced settings (score threshold, hybrid alpha)
- [ ] Submit and verify profile appears in list

### Edit Profile
- [ ] Open existing profile for editing
- [ ] Modify name and verify it updates
- [ ] Change collections and save
- [ ] Toggle enabled/disabled status
- [ ] Verify changes persist after page refresh

### Delete Profile
- [ ] Delete a profile
- [ ] Verify it's removed from the list
- [ ] Verify deleted profile no longer appears in MCP tools

## MCP Server

### Connection
- [ ] Start MCP server with `semantik-mcp serve --profile <name>`
- [ ] Verify server starts without errors
- [ ] Verify server logs show connection to WebUI

### Profile Filtering
- [ ] Start server with `--profile <name>` flag
- [ ] Verify only specified profile's search tool is exposed
- [ ] Start server without filter, verify all enabled profiles exposed

### Tool Listing
- [ ] Request list_tools from MCP client
- [ ] Verify `search_<profile>` tool appears for each enabled profile
- [ ] Verify utility tools appear: get_document, get_document_content, get_chunk, list_documents, diagnostics

## Search Functionality

### Basic Search
- [ ] Execute search via `search_<profile>` tool
- [ ] Verify results are returned
- [ ] Verify result format includes: collection_id, document_id, chunk_id, score, text

### Search Parameters
- [ ] Override k (result count)
- [ ] Override search_type
- [ ] Toggle use_reranker
- [ ] Set score_threshold and verify filtering works
- [ ] Set hybrid_alpha for hybrid search

### Cross-Profile Scope
- [ ] Attempt to access document from non-scoped collection
- [ ] Verify access is denied with appropriate error

## Utility Tools

### get_document
- [ ] Get document metadata by ID
- [ ] Verify metadata includes: filename, size, status, chunk_count

### get_document_content
- [ ] Get full document content
- [ ] Verify text content is returned
- [ ] Verify binary documents return appropriate error

### get_chunk
- [ ] Get specific chunk by ID
- [ ] Verify chunk text is returned

### list_documents
- [ ] List documents in a collection
- [ ] Test pagination (page, per_page)
- [ ] Test status filter

### diagnostics
- [ ] Call diagnostics tool
- [ ] Verify server_name, webui_url are correct
- [ ] Verify connection.connected is true
- [ ] Verify connection.authenticated is true
- [ ] Verify profiles list shows expected profiles
- [ ] Verify cache status is reported

## Error Handling

### Authentication Errors
- [ ] Start server with invalid token
- [ ] Verify appropriate error message
- [ ] Verify diagnostics shows authentication failed

### Connection Errors
- [ ] Start server with wrong WebUI URL
- [ ] Verify appropriate error message
- [ ] Verify diagnostics shows connection failed

### Profile Errors
- [ ] Disable a profile while server is running
- [ ] Verify search fails with "profile not found" after cache expires
- [ ] Request non-existent profile
- [ ] Verify appropriate error message

## Claude Desktop Integration

### Setup
- [ ] Add Semantik to claude_desktop_config.json
- [ ] Restart Claude Desktop
- [ ] Verify Semantik appears in available tools

### Usage
- [ ] Ask Claude to search your documents
- [ ] Verify Claude uses the search tool
- [ ] Verify results are incorporated into response
- [ ] Ask Claude to get document details
- [ ] Ask Claude to use diagnostics for debugging

## Performance

- [ ] Search returns results in reasonable time (<5s)
- [ ] Multiple rapid requests don't cause errors
- [ ] Server handles reconnection after temporary disconnection

## Logging

- [ ] Verify INFO level logs show tool calls
- [ ] Enable verbose mode (`-v`) and verify DEBUG logs
- [ ] Verify search duration is logged
- [ ] Verify cache hits/misses are logged at DEBUG level

## Edge Cases

- [ ] Profile with no collections (should show error)
- [ ] Search with empty query (should show error)
- [ ] Very long search query
- [ ] Special characters in search query
- [ ] Unicode in search query
- [ ] Collection with no indexed documents

## Cleanup

- [ ] Delete test profiles
- [ ] Stop MCP server
- [ ] Verify no orphaned processes

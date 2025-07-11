"""Packages directory for the document embedding system.

This directory contains three main packages:

- vecpipe: Document processing and vector embedding pipeline
- webui: Web API and user interface backend
- shared: Common utilities, database, and shared components

The shared package ensures proper decoupling between vecpipe and webui,
allowing them to operate independently without circular dependencies.
"""

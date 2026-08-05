#!/usr/bin/env python
"""
Server startup script.
Run from project root: python start_server.py
"""
import sys
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("  HiveMind - Universal Multi-Agent Orchestrator")
    print("="*60)
    print("  Backend: http://localhost:8000")
    print("  Docs:    http://localhost:8000/docs")
    print("  Health:  http://localhost:8000/health")
    print("="*60 + "\n")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(BACKEND_ROOT)],
    )

#!/usr/bin/env python3
"""
EZ Claim React Runner
Launches the FastAPI backend and serves the React application interface.
"""
import os
import sys
import subprocess
import uvicorn

def main():
    print("=" * 60)
    print("🚀 EZ Claim Pipeline - React Web Application Launcher")
    print("=" * 60)

    # Check if frontend dist folder exists
    dist_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")

    if not os.path.exists(dist_path):
        print("\n📦 Building React frontend production bundle...")
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        try:
            # Install npm packages if node_modules missing
            node_modules = os.path.join(frontend_dir, "node_modules")
            if not os.path.exists(node_modules):
                print(" Running 'npm install' in frontend directory...")
                subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

            print(" Running 'npm run build' in frontend directory...")
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
            print("✓ React application successfully compiled into production bundle.")
        except Exception as e:
            print(f"⚠️  Frontend compilation note: {e}")
            print(" Starting backend API server on http://localhost:8000")

    print("\n🌐 Starting Server at: http://localhost:8000")
    print("Press Ctrl+C to stop the server.\n")

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

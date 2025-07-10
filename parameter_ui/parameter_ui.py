#!/usr/bin/env python3
# filepath: /global/homes/a/ah1032/bin/gkyl-ui.py
import os
import sys
import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path

def find_ui_directory():
    """Find the parameter_ui directory"""
    script_dir = Path(__file__).parent
    ui_dir = script_dir.parent / "personal_gkyl_scripts" / "parameter_ui"
    
    if ui_dir.exists():
        return ui_dir
    
    # Alternative search locations
    home_dir = Path.home()
    for search_path in [
        home_dir / "personal_gkyl_scripts" / "parameter_ui",
        Path("/global/homes/a/ah1032/personal_gkyl_scripts/parameter_ui"),
    ]:
        if search_path.exists():
            return search_path
    
    return None

def start_server(port=8000):
    ui_dir = find_ui_directory()
    if not ui_dir:
        print("Error: Could not find parameter_ui directory")
        sys.exit(1)
    
    os.chdir(ui_dir)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress server logs
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Starting Gkyl Parameter UI on port {port}")
        print(f"Serving from: {ui_dir}")
        print(f"Open http://localhost:{port} in your browser")
        
        # Auto-open browser after a short delay
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: gkyl-ui.py [port]")
            sys.exit(1)
    
    start_server(port)
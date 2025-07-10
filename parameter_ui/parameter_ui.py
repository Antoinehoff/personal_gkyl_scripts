#!/usr/bin/env python3
# filepath: /global/homes/a/ah1032/personal_gkyl_scripts/parameter_ui/parameter_ui.py
import os
import sys
import http.server
import socketserver
import webbrowser
import threading
import time
import socket
from pathlib import Path

def find_ui_directory():
    """Find the parameter_ui directory"""
    script_dir = Path(__file__).parent
    ui_dir = script_dir  # Since script is in parameter_ui directory
    
    if (ui_dir / "index.html").exists():
        return ui_dir
    
    # Alternative search locations
    home_dir = Path.home()
    for search_path in [
        home_dir / "personal_gkyl_scripts" / "parameter_ui",
        Path("/global/homes/a/ah1032/personal_gkyl_scripts/parameter_ui"),
    ]:
        if search_path.exists() and (search_path / "index.html").exists():
            return search_path
    
    return None

def find_free_port(start_port=8000):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError("No free ports found")

def start_server(requested_port=8000):
    ui_dir = find_ui_directory()
    if not ui_dir:
        print("Error: Could not find parameter_ui directory with index.html")
        sys.exit(1)
    
    os.chdir(ui_dir)
    
    # Try to find an available port
    try:
        port = find_free_port(requested_port)
        if port != requested_port:
            print(f"Port {requested_port} is busy, using port {port} instead")
    except OSError:
        print("Error: Could not find an available port")
        sys.exit(1)
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Only log errors, not every request
            if "error" in format.lower():
                super().log_message(format, *args)
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Starting Gkyl Parameter UI on port {port}")
            print(f"Serving from: {ui_dir}")
            print(f"Open http://localhost:{port} in your browser")
            print("Press Ctrl+C to stop the server")
            
            # Auto-open browser after a short delay
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: parameter_ui.py [port]")
            sys.exit(1)
    
    start_server(port)
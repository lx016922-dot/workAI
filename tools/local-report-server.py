"""Stream one explicitly selected report over loopback, never a directory.

Avoid sending large embedded fields through the browser automation protocol.
Callers block every other browser request and disable networking after loading.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import shutil
from threading import Thread

@contextmanager
def serve_report(path):
    report=Path(path).resolve(strict=True)
    if not report.is_file():raise ValueError('A report file is required.')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/report':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(report.stat().st_size));self.send_header('Cache-Control','no-store');self.end_headers()
            with report.open('rb') as file:shutil.copyfileobj(file,self.wfile,1024*1024)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(server.server_port)+'/report'
    finally:server.shutdown();server.server_close();thread.join()

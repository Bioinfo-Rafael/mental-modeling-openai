"""Local viewer server, serving exact decoded recording frames."""
import argparse
from functools import partial
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import time
from urllib.parse import urlsplit

from prepare import ROOT, prepare


class ViewerHandler(SimpleHTTPRequestHandler):
    """Always reload viewer code and JSON; keep decoded frame caching intact."""

    def is_viewer_asset(self):
        return Path(urlsplit(self.path).path).suffix in {'.html', '.js', '.css', '.json'} or urlsplit(self.path).path.endswith('/')

    def send_head(self):
        if self.is_viewer_asset():
            # A cached pre-upgrade response may still send a conditional request.
            for header in ('If-Modified-Since', 'If-None-Match'):
                if header in self.headers:
                    del self.headers[header]
        return super().send_head()

    def end_headers(self):
        if self.is_viewer_asset():
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8130)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    prepare()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(ViewerHandler, directory=str(ROOT)))
    url = f'http://127.0.0.1:{server.server_port}/viewer/?v={time.time_ns()}'
    print(f'LAG viewer: {url}', flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

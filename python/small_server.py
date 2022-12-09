import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from VideoInputParser import VideoInputParser
from urllib.parse import parse_qs
import cgi


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/analyze-video':
            try:
                # Parse the video file from the POST request
                # video_file = self.rfile.read(
                #     int(self.headers['Content-Length']))
                # video_bytes = bytearray(video_file)

                # Parse the request data using cgi.FieldStorage
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type'],
                    }
                )

                # Get the "video" file from the request data
                video_file = form['video'].file
                video_bytes = bytearray(video_file.read())

                vip = VideoInputParser()
                result = vip.parse_video(video_bytes)

                # Convert the result to a JSON string
                json_str = json.dumps(result)

                # Return a 200 status code and the result as a JSON object
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(bytes(json_str, 'utf-8'))
            except Exception as e:
                print('except', e)
                # If the parse function throws an error, return a 400 status code
                self.send_response(400)
        else:
            self.send_response(404)


def run():
    httpd = HTTPServer(('localhost', 8000), RequestHandler)
    print("Server hosting on port 8000")
    httpd.serve_forever()


run()

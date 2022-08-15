import socketserver

class handler(socketserver.BaseRequestHandler):

    def handle(self):
        """
        action on client request
        """

        data = self.request.recv(1024).strip()
        client_addr = self.client_address

        print("msg from {}: {}".format(str(client_addr), data))
        self.request.sendall(data.upper())


HOST, PORT = "127.0.0.1", 9999

server = socketserver.TCPServer((HOST, PORT), handler)

server.serve_forever()

import socketserver
import socket

host, port = "127.0.0.1", 9999
buffersize = 1024 # 1g

class TCPhandler(socketserver.BaseRequestHandler):

    def handle(self):
        """
        action on client request
        """

        data = self.request.recv(1024).strip()
        client_addr = self.client_address

        print("msg from {}: {}".format(str(client_addr), data))
        self.request.sendall(data.upper())



def spin_tcp_server(HOST, PORT):

    server = socketserver.TCPServer((HOST, PORT), TCPhandler)

    server.serve_forever()

udpsocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

udpsocket.bind((host, port))

while(True):
    packet = udpsocket.recvfrom(buffersize)
    msg = str(packet[0])
    address = packet[1][0]
    print("received packet from {}".format(address))
    print("msg: {}".format(msg))

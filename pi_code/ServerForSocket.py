import socket
TCP_IP = "100.84.40.82"
TCP_PORT = 3000
BUFFER_SIZE = 20 #NOrmally 1024 but we want a fast response
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
conn, addr = s.accept()
print("Connction address: ", addr)
while 1:
  data = conn.recv(BUFFER_SIZE)
  if not data: break
  print("received data: ", data)
conn.close()
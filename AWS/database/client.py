import socket
import json
from datetime import datetime 

print("We're in tcp client...")
#the server name and port client wishes to access
server_name = '13.41.53.180'
#'52.205.252.164'
server_port = 12000
#create a TCP client socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#Set up a TCP connection with the server
#connection_socket will be assigned to this client on the server side
client_socket.connect((server_name, server_port))
# create a sample JSON object
data = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
    "accel_x": 90.1,
    "accel_y": 87.21,
    "accel_z": 0.0001
}
# encode JSON object to bytes
json_data = json.dumps(data).encode()

# send the JSON data to the server
client_socket.send(json_data)

#return values from the server
msg = client_socket.recv(1024)
print(msg.decode())
client_socket.close()
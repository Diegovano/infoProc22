import socket
import threading
#select a server port
server_port = 12000
connections = []
nicknames = []
#create a welcoming socket
welcome_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#bind the server to the localhost at port server_port
print("Waiting for Clients...")
welcome_socket.bind(('0.0.0.0',server_port))
welcome_socket.listen(5)
#handle multiple clients
#ready message
print('Server running on port ', server_port) 
test = "joined"

def leave(connection,nickname):
	connections.remove(connection)
	connection.close()
	broadcast(nickname + " left!")
	nicknames.remove(nickname)

def broadcast(message):
	for conection in connections:
		conection.send(message.encode())

def threaded_client(connection,nickname):
	while True:
		try:
			#send message
			data = connection.recv(1024).decode()
			if data != "exit":
				data = (nickname + ": " + data) 
				broadcast(data)
			else: 
				connection.send("Server: Goodbye, see you soon :)".encode())
				leave(connection,nickname)
				break
		except:
			leave(connection,nickname)
			break

#Now the main server loop
while True:
	#notice recv and send instead of recvto and sendto
	Client, address = welcome_socket.accept()
	Client.send(test.encode())
	nickname = Client.recv(1024).decode()
	nicknames.append(nickname)
	connections.append(Client)
	print("Nickname is: " + nickname)
	Client.send(("Connected to server!\nHi "+nickname+" use \"exit\" to leave have fun ;)").encode())
	broadcast(nickname + " Joined!")
	thread = threading.Thread(target=threaded_client,args=(Client,nickname,))
	thread.start()

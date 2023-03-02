import psutil
import socket

# Define the IP and port number to close sockets on
ip_address = '13.41.53.180'
port_number = 12000

# Get a list of all processes with open sockets
processes = []
for p in psutil.net_connections():
    if p.status == psutil.CONN_ESTABLISHED and p.laddr.ip == ip_address and p.laddr.port == port_number:
        processes.append(p)

# Close all sockets by terminating the associated processes
for p in processes:
    print(f"Closing socket {p.laddr.ip}:{p.laddr.port} for PID {p.pid}")
    psutil.Process(p.pid).terminate()


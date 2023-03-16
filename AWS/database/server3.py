import socket
import json
import boto3
from pprint import pprint
from datetime import datetime
import decimal 
import select


def create_table(table_name, dynamodb = None):
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    
    # Check if table already exists
    table_exists = False
    for table in dynamodb.tables.all():
        if table.name == table_name:
            print(f"Table '{table_name}' already exists")
            return table
    # If table does not exist, create it
    if not table_exists:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'timestamp',
                    'KeyType': 'HASH'  #Partition Key
                },
                {
                     'AttributeName': 'device_id',
                     'KeyType': 'RANGE'  #Sort key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'timestamp',
                    'AttributeType': 'S'
                 },
                 {
                     'AttributeName': 'device_id',
                     'AttributeType': 'S'
                 }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )

        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Table '{table_name}' created successfully")

        return table

def add_item(table_name, input_json, dynamodb=None):
    """ input json should be in the format of
    {
        "timestamp": "YYYY-MM-DD HH:MM:SS.ssssss",
        "device_id" : <string>
        "change_step": <float>,
        "direction": <float>,
    }
    """
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(table_name)

    try:
        input_dict = json.loads(input_json)
        input_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
             
    except json.JSONDecodeError:
        return f"Error: invalid input JSON format"
     
    # Check that required keys are present in the input dictionary
    required_keys = ['timestamp', 'device_id','change_step', 'heading']
    if not all(key in input_dict for key in required_keys):
        return f"Error: missing one or more required keys: {required_keys}"

    # Parse datetime string into datetime object
    try:
        timestamp = datetime.strptime(input_dict['timestamp'], '%Y-%m-%d %H:%M:%S.%f')

    except ValueError:
        return f"Error: invalid timestamp format"
   
    # Construct the item to be added to the table
    item = {
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
        'device_id' : str(input_dict['device_id']),
        'change_step' : decimal.Decimal(str(input_dict['change_step'])),
        'heading' : decimal.Decimal(str(input_dict['heading'])),
        }

    # Add the item to the table
    response = table.put_item(Item=item)
    return response

#to do : make decimal of acceleration to 8 bits, get displacement from the database algorithm

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'directionn_data'
table = create_table(table_name, dynamodb)

#select a server port
server_port = 12000
welcome_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
welcome_socket.bind(('0.0.0.0', server_port))
welcome_socket.listen()
print('Server running on port', server_port)

sockets = [welcome_socket]  # list of sockets to monitor
clients = {}  # dictionary of client sockets and their addresses

while True:
    readable, _, _ = select.select(sockets, [], [])  # wait for a socket to be ready to read
    for sock in readable:
        if sock is welcome_socket:  # new client connection
            connection_socket, caddr = sock.accept()
            sockets.append(connection_socket)
            clients[connection_socket] = caddr
            print('New client connected:', caddr)
        else:  # existing client data received
            cmsg = sock.recv(1024)
            if not cmsg:  # client disconnected
                sock.close()
                sockets.remove(sock)
                del clients[sock]
                print('Client disconnected:', clients[sock])
            else:  # process client data
                decoded_data = json.loads(cmsg.decode())
                print('Received from', clients[sock], ':', decoded_data)
                response = add_item(table_name, json.dumps(decoded_data), dynamodb)
               

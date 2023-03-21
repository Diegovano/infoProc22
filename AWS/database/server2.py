import socket
import json
import boto3
from pprint import pprint
from datetime import datetime
import decimal 
import threading

# for a given device, get 5 consecutive samples and make them to a numpy arraay,
#  where the first column is the timestamp, the second column is the total steps, 
# third is direction
#keep sending sets of 5 at a time

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
                     'AttributeName': 'device_id',
                     'KeyType': 'HASH'  #Partition key
                },
                {
                    'AttributeName': 'timestamp',
                    'KeyType': 'RANGE'  #Sort Key
                },
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
    
def create_table_basic(table_name, dynamodb = None):
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
                     'AttributeName': 'device_id',
                     'KeyType': 'HASH'  #Partition key
                }
            ],
            AttributeDefinitions=[
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
         "change_step": <int>,
         "direction": <int>,
    }
    """
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(table_name)

    try:
        input_dict = json.loads(input_json)
             
    except json.JSONDecodeError:  
        return f"Error: invalid input JSON format"
     
    required_keys = ['timestamp', 'device_id','total_steps', 'heading']
    if not all(key in input_dict for key in required_keys):
        return f"Error: missing one or more required keys: {required_keys}"

    item = {
        'timestamp': input_dict['timestamp'], # Εβριθινγ ισ ουκινγ
        'device_id' : input_dict['device_id'], #ζηε δθαν δαι μα ται ψηθν
        'total_steps' :decimal.Decimal(str(input_dict['total_steps'])),
        'heading' : decimal.Decimal(str(input_dict['heading'])),
        'x-cordinate': 'x-cordinate' ,
        'y-cordinate': 'y-cordinate',
        }

    response = table.put_item(Item=item)
    return response

def add_item_basic(table_name, input_dict, dynamodb=None):
    """ input json should be in the format of
    {
        "device_id" : <string>
    }
    """
    if not dynamodb:
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(table_name)
     
    # Check that required keys are present in the input dictionary
    required_keys = ['device_id']
    if not all(key in input_dict for key in required_keys):
        return f"Error: missing one or more required keys: {required_keys}"

    # Construct the item to be added to the table
    item = {
        'device_id' : input_dict['device_id'], #ζηε δθαν δαι μα ται ψηθν
        }

    # Add the item to the table
    response = table.put_item(Item=item)
    return response

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'di_data10'
table = create_table(table_name, dynamodb)
DeviceIDs = create_table_basic('DeviceIds', dynamodb)
def last_total_steps(device_id):
    dynamodb = boto3.client('dynamodb', region_name='us-east-1')
    # device_id to query (so far either 1 or 2) 
    # set up the query parameters
    query_params = {
        'TableName': 'di_data10',
        'KeyConditionExpression': 'device_id = :device_id',
        'ExpressionAttributeValues': {
            ':device_id': {'S': device_id},
        },
        'ScanIndexForward': False,
        'Limit': 1
    }
    response = dynamodb.query(**query_params)

    if len(response['Items']) > 0:
        last_total_steps = int(response['Items'][0]['total_steps']['N'])
        print(last_total_steps)
    else:
        print("No data found for the specified device id.")

#select a server port
server_port = 12000
#create a welcoming socket
connections = []
nicknames = {}
welcome_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
welcome_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#bind the server to the localhost at port server_port
welcome_socket.bind(('0.0.0.0',server_port))
welcome_socket.listen(5)
def leave(connection,nickname):
    connections.remove(connection)
    connection.close()
    broadcast(nickname + " left!")
    nicknames.remove(nickname)

def broadcast(message):
    for conection in connections:
        conection.send(message.encode())

def threaded_client(connections):
    nickname = ""
    while True:
                cmsg = connections.recv(128)
                if not cmsg:
                    break
                messages = cmsg.split(b"\n")
                for message in messages:
                    message = message.strip()
                    if message:
                        print("Decoding",message)
                        try:
                            decoded_data = json.loads(message.decode())
                            print(decoded_data)
                            add_item(table_name, json.dumps(decoded_data), dynamodb)
                            nickname = decoded_data["device_id"]
                            if nicknames.get(nickname, None) is None:
                                print("Adding nickname to db "+nickname)
                                add_item_basic('DeviceIds',{"device_id":nickname},dynamodb)
                            nicknames[nickname] = last_total_steps(nickname)
                            
                            nick_list = list(nicknames.items())  # = [[key, value], [key, value]]
                            nick_list.sort(key=lambda pair: pair[1])
                            index = -1
                            for  i in range(len(nick_list)):
                                nick_pair = nick_list[i]
                                if nick_pair[0] == nickname:
                                     index = i
                            if index == -1:
                                 response_msg="69".encode()
                                 print("shit balls")
                            else:
                                response_msg = str(index+1).encode()
                            connections.send(response_msg)
                            #print(response)
                        except json.decoder.JSONDecodeError:
                                print("Failed to decode!")

#Now the main server loop
try:
        while True:
            #notice recv and send instead of recvto and sendto
            Client, address = welcome_socket.accept()
            #first_send = json.load(Client.recv(1024).decode())
            #nickname = first_send['device_id']
            #add_item_basic('DeviceIds',{"device_id":nickname},dynamodb)
            #nicknames[nickname] = 0
            connections.append(Client)
            #print("Device ID: " + nickname)
            Client.send('c'.encode())
            thread = threading.Thread(target=threaded_client,args=(Client,))
            thread.start()
except KeyboardInterrupt:
   welcome_socket.close()


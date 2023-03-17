import socket
import json
import boto3
from pprint import pprint
from datetime import datetime
import decimal 

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
     
    # Check that required keys are present in the input dictionary
    required_keys = ['timestamp', 'device_id','total_steps', 'heading']
    if not all(key in input_dict for key in required_keys):
        return f"Error: missing one or more required keys: {required_keys}"

    # Construct the item to be added to the table
    item = {
        'timestamp': input_dict['timestamp'], # Εβριθινγ ισ ουκινγ
        'device_id' : input_dict['device_id'], #ζηε δθαν δαι μα ται ψηθν
        'total_steps' :decimal.Decimal(str(input_dict['total_steps'])),
        'heading' : decimal.Decimal(str(input_dict['heading'])),
        }

    # Add the item to the table
    response = table.put_item(Item=item)
    return response

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'di_data10'
table = create_table(table_name, dynamodb)

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = 50000                # Reserve a port for your service.

print 'Server started!'
print 'Waiting for clients...'

s.bind((host, port))        # Bind to the port
s.listen(5)                 # Now wait for client connection.
c, addr = s.accept()     # Establish connection with client.
print 'Got connection from', addr
while True:
   msg = c.recv(1024)
   print addr, ' >> ', msg
   msg = raw_input('SERVER >> ')
   c.send(msg);
   #c.close()     
#select a server port
server_port = 12000
#create a welcoming socket
welcome_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
welcome_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#bind the server to the localhost at port server_port
welcome_socket.bind(('0.0.0.0',server_port))
welcome_socket.listen()
#ready message
print('Server running on port ', server_port)

connection_socket, caddr = welcome_socket.accept()
#Now the main server loop
try: 
    while True:
        cmsg = connection_socket.recv(1024)
        print(cmsg)
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
                    response = add_item(table_name, json.dumps(decoded_data), dynamodb)
                    # print('ELLADA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    response_msg = str(response).encode()
                    connection_socket.send(response_msg)
                    response_msg = 'c'.encode()
                    connection_socket.send(response_msg)
                    #print(response)
                except json.decoder.JSONDecodeError:
                    print("Failed to decode!")

except KeyboardInterrupt:
    connection_socket.close()
    welcome_socket.close()

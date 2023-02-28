import socket
import json
import boto3
from pprint import pprint
from datetime import datetime
import decimal 

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
                    'KeyType': 'HASH'
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'timestamp',
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
        "accel_x": <float>,
        "accel_y": <float>,
        "accel_z": <float>
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
    required_keys = ['timestamp', 'accel_x', 'accel_y', 'accel_z']
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
    
        'accel_x' : decimal.Decimal(str(input_dict['accel_x'])),
        'accel_y' : decimal.Decimal(str(input_dict['accel_y'])),
        'accel_z' : decimal.Decimal(str(input_dict['accel_z']))
        }

    # Add the item to the table
    response = table.put_item(Item=item)
    return response


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'accel_data'
table = create_table(table_name, dynamodb)

#select a server port
server_port = 12000
#create a welcoming socket
welcome_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
#bind the server to the localhost at port server_port
welcome_socket.bind(('0.0.0.0',server_port))
welcome_socket.listen(1)
#ready message
print('Server running on port ', server_port)

connection_socket, caddr = welcome_socket.accept()
#Now the main server loop
try: 
    while True:
        #notice recv and send instead of recvto and sendto
        cmsg = connection_socket.recv(1024)  
        if not cmsg:
            break
        # decode json data received
        decoded_data = json.loads(cmsg.decode())
        
        # print the received data
        print("Received:", decoded_data)

        # add the incoming data to the DynamoDB table
        response = add_item(table_name, json.dumps(decoded_data), dynamodb)

except KeyboardInterrupt:
    connection_socket.close()
    welcome_socket.close()

   

import json
import socket
import ssl
from datetime import datetime
import decimal
import boto3


def create_table(table_name, dynamodb):
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'timestamp',
                'KeyType': 'HASH'
            },
            {
                'AttributeName': 'device_id',
                'KeyType': 'RANGE'
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
            'ReadCapacityUnits': 1,
            'WriteCapacityUnits': 1
        }
    )

    # Wait until the table exists.
    table.meta.client.get_waiter('table_exists').wait(TableName=table_name)

    return table


def add_item(table_name, input_json, dynamodb):
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
    response = dynamodb.Table(table_name).put_item(Item=item)
    return response


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'dir_data'
table = create_table(table_name, dynamodb)

# select a server port
server_port = 12000
# create a welcoming socket
welcome_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bind the server to the localhost at port server_port
welcome_socket.bind(('0.0.0.0', server_port))
welcome_socket.listen()
# ready message
print('Server running on port ', server_port)

connection_socket, caddr = welcome_socket.accept()
# Now the main server loop
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

       

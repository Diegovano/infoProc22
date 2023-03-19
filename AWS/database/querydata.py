import boto3
import numpy as np

dynamodb = boto3.client('dynamodb', region_name='us-east-1')

# device_id to query (so far either 1 or 2) 
device_id = '1'

# set up the query parameters
query_params = {
    'TableName': 'di_data10',
    'KeyConditionExpression': 'device_id = :device_id',
    'ExpressionAttributeValues': {
        ':device_id': {'S': device_id},
    },
}

data = []
#can add timestamp as well, prob not needed 
response = dynamodb.query(**query_params)
for item in response['Items']:
    heading = int(item['heading']['N'])
    total_steps = int(item['total_steps']['N'])
    data.append([total_steps, heading])
    if len(data) == 5:
        data_array = np.array(data)
        print(data_array)
        data.clear()

if len(data) > 0:
    data_array = np.array(data)
    print(data_array)


import boto3

import boto3

def last_total_steps(device_id):
    dynamodb = boto3.client('dynamodb', region_name='us-east-1')
    # device_id to query (so far either 1 or 2) 
    device_id = 'Cozzy'
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



device_id = 'Cozzy'
get_last_steps = last_total_steps(device_id)
print(last_total_steps)



from pprint import pprint
import boto3
from boto3.dynamodb.conditions import Key


dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'di_data10'
table = dynamodb.Table(table_name)
    #query data etc
response = table.query(
     KeyConditionExpression = 'device_id = 1',
    ProjectionExpression='Heading, #cs, #ds, #sm',
    ExpressionAttributeNames={
        '#cs': 'total_steps',
        '#sm' : 'heading',
         '#ds': 'timestamp',
        })
items = response['Items']
print (response)
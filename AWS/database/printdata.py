import boto3

# Create a DynamoDB resource
dynamodb = boto3.resource('dynamodb' , region_name='us-east-1')

# Get a reference to the table
table = dynamodb.Table('di_data5')

# Scan the table and print each item
response = table.scan()
items = response['Items']
for item in items:
    print(item)

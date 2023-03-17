import boto3 
from flask import Flask, make_response, jsonify, request

app = Flask("application")
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table_name = 'di_data10'
table = dynamodb.Table(table_name)

@app.route('/get_info', methods=['GET'])
def get_info():
    args = request.args
    if not "device_id" in args:
        return make_response(jsonify({'error': 'Missing Parameter'}), 400) 
    #query data etc
    response = table.query(
        KeyConditionExpression = 'device_id = :id_val',
        ExpressionAttributeValues={
            ':id_val': args["device_id"],
        },
        ProjectionExpression='Heading, #cs, #ds, #sm',
        ExpressionAttributeNames={
            '#cs': 'total_steps',
            '#sm' : 'heading',
            '#ds': 'timestamp',
        },
    )
    items = response['Items']
    http_resp = make_response(jsonify({'Items' : items}), 200)
    http_resp.headers.add('Access-Control-Allow-Origin', '*')
    return http_resp
# //http://ip:port/get_info?device_id=1

if __name__ == '__main__':
    app.run(debug = True, host = "0.0.0.0", port= 8080)
    
# timestamp, total steps and device id, heading 

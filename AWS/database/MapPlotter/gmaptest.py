import gmplot
import time
import datagenerator as dg
import boto3 


apikey = 'AIzaSyAnqGZhAKKw4ZFbCi_0shUYXGHR0XFsWsg' # (your API key here)
gmap = gmplot.GoogleMapPlotter(51.49920108221579, -0.17424716105896274, 23, apikey=apikey)

def get_accel_data ():
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('accel_data') 
    response = table.scan()
    items = response['Items']
    data = []
    for item in items:
        x_value = item[accel_x]
        y_value = item[accel_y]
    data.append((x_value, y_value)) 
    return data

coords = [
    (51.49875113729699, -0.17692285989733622),
    (51.49875113729699, -0.17692285989733622),
    (51.49875113729699, -0.17692285989733622),
    (51.49875113729699, -0.17692285989733622),
    (51.49875113729699, -0.17692285989733622),
    (51.49875113729699, -0.17692285989733622)
]
def find_n(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start

for i in range(200):
    time.sleep(1)
    h,v = dg.increment((51.51157014690242, -0.19124352334865424),(51.50249343996652, -0.15188949941867913),time.time())
    coords.append((coords[5][0]+h,coords[5][1]+v))
    coords.pop(0)
    print(coords)
    path = zip(*coords)
    gmap = gmplot.GoogleMapPlotter(coords[5][0], coords[5][1], 20, apikey=apikey)
    gmap.plot(*path, edge_width=2, color='red')
    a = gmap.get()
    b = find_n(a,'\n',2)
    c = a[:b] + '\n<meta http-equiv=\"refresh\" content=\"1\">' + a[b:]
    #print(a)
    #print(c)
    with open("test.html", "w") as file:
        file.write(c)

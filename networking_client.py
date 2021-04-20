import random
import zmq
import time
import json

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5556")

while True:
    configurations = [random.randint(0, 359), random.randint(0, 359)]
    request = json.dumps(configurations)
    socket.send(request.encode("UTF-8"))
    response = json.loads(socket.recv())

    if response["status"] == 2:
        print("Error")
    else:
        print("Success")

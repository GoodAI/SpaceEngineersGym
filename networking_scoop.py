import random
import zmq
import time
import json
from scoop import futures


def run(id):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")

    while True:
        configurations = [random.randint(0, 360) for _ in range(5)]
        request = json.dumps(configurations)
        socket.send(request.encode("UTF-8"))
        response = json.loads(socket.recv())

        if response["status"] == 2:
            print(f"{id}: Error")
        else:
            print(f"{id}: Success")

        # print(response)


if __name__ == '__main__':
    clients = range(0, 25)
    results = list(futures.map(run, clients))


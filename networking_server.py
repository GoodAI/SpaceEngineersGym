import zmq

context = zmq.Context()
socket = context.socket(zmq.ROUTER)
socket.bind("tcp://*:5556")

for i in range(1000):
    rep = socket.recv_multipart()  # This blocks until we get something
    print('Ping got reply:', rep)
import json
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5562")


class AgentController:
    """
    Simple agent controller for Space Engineers.
    It connects to the iv4xr-plugin-se via TCP/IP socket to control the game.
    """
    def __init__(self):
        self.id = None
        self._send_initial_request()

    def _send_command(self, command):
        request = {
            "id": self.id,
            "type": "Command",
            "command": command,
        }

        response = self._send_request(request)

        return response

    def _send_request(self, request):
        request_message = json.dumps(request)
        socket.send(request_message.encode("UTF-8"))
        response = json.loads(socket.recv())

        return response

    def move_forward(self, num_frames=1):
        return self.move(MoveArgs(0, 0, -10), num_frames=num_frames)

    def move(self, move_direction, jump=False, num_frames=1):
        for _ in range(num_frames):
            observation = self._send_command({
                "type": "Move",
                "move": vars(move_direction),
            })

        return observation

    def teleport(self, position):
        observation = self._send_command({
            "type": "Teleport",
            "position": vars(position),
        })

        return observation

    def observe(self):
        return self._send_command({
            "type": "Observe",
        })

    def close_connection(self):
        if self.id is not None:
            request = {
                "type": "Stop",
                "id": self.id,
            }
            self._send_request(request)

    def _send_initial_request(self):
        request = {
            "type": "Initial",
        }
        response = self._send_request(request)
        self.id = response["id"]


class MoveArgs:
    def __init__(self, x, y, z):
        self.X = x
        self.Y = y
        self.Z = z

    def get_serializable(self):
        return {
            "X": str(self.X),
            "Y": str(self.Y),
            "Z": str(self.Z),
        }


if __name__ == '__main__':
    agent = AgentController()

    print("Do nothing, just observe and print the observation")
    observation = agent.observe()
    print(observation)

    print("Move forward for 50 frames and print the observation")
    observation = agent.move_forward(50)
    print(observation)

    print("Teleport")
    observation = agent.teleport(MoveArgs(-510.71, 379, 385.20))
    print(observation)

    agent.close_connection()

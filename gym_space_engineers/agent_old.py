import json
import socket

SERVER_IP = "localhost"
SERVER_PORT = 9678


class AgentController:
    """
    Simple agent controller for Space Engineers.
    It connects to the iv4xr-plugin-se via TCP/IP socket to control the game.
    """

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((SERVER_IP, SERVER_PORT))

    def _send_command(self, command):
        message = {
            "Cmd": "AGENTCOMMAND",
            "Arg": command,
        }
        message_serialized = json.dumps(message, separators=(",", ":"))
        self.socket.sendall(bytes(message_serialized + "\n", "utf-8"))

        return self._get_observation()

    def _get_observation(self):
        data_bytes = self.socket.recv(100000)
        data_str = data_bytes.decode("UTF-8")
        observation = json.loads(data_str)

        return observation

    def move_forward(self, num_frames=1):
        return self.move(MoveArgs(0, 0, -1), num_frames=num_frames)

    def move(self, move_direction, jump=False, num_frames=1):
        for i in range(num_frames):
            observation = self._send_command(
                {
                    "Cmd": "MOVETOWARD",
                    "Arg": {
                        "Object1": vars(move_direction),
                        "Object2": jump,
                    },
                }
            )

        return observation

    def rotate(self, rotation, num_frames=1):
        for i in range(num_frames):
            observation = self._send_command(
                {
                    "Cmd": "MOVE_ROTATE",
                    "Arg": {
                        "Rotation3": vars(rotation),
                    },
                }
            )

        return observation

    def teleport(self, position):
        observation = self._send_command(
            {
                "Cmd": "TELEPORT",
                "Arg": {
                    "Object1": vars(position),
                },
            }
        )

        return observation

    def observe(self):
        return self._send_command(
            {
                "Cmd": "OBSERVE",
            }
        )

    def close_connection(self):
        self.socket.close()


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


if __name__ == "__main__":
    agent = AgentController()

    print("Do nothing, just observe and print the observation")
    observation = agent.observe()
    print(observation)

    print("Move forward for 50 frames and print the observation")
    observation = agent.move_forward(50)
    print(observation)

    print("Rotate for 50 frames and print the observation")
    observation = agent.rotate(MoveArgs(0, 50, 0), num_frames=50)
    print(observation)

    print("Teleport")
    observation = agent.teleport(MoveArgs(500, 500, 500))
    print(observation)

    agent.close_connection()

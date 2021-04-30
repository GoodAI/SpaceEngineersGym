# SpaceEngineersGym

This repository contains an OpenAI Gym environment which makes it possible to control agents/robots in Space Engineers. This repository is currently in development and requires an additional Space Engineers plugin (which is currently not public) to run the code.

## Installation

- Download Space Engineers from [Steam](https://store.steampowered.com/app/244850/Space_Engineers/) and install the game (primarily for Windows)
- Download the VeriDream binaries zip (these binaries are provided on request)
- Put the *VeriDream* folder from the zip into *Steam\steamapps\common\SpaceEngineers\Bin64*
- IMPORTANT: Make sure Windows is OK to run the libraries in the *VeriDream/bin* folder. Windows 10 blocks "randomly" downloaded libraries. To unblock them, right-click each of them and open file properties. Look for Security section on the bottom part of the General tab. You might see a message: "This file came from another computer and might be blocked...". If so, check the Unblock checkbox. (If you skip this step, the game will probably crash with a message: System.NotSupportedException: An attempt was made to load an assembly from a network location...)
- Install this module with `pip3 install .` (use -e flag if you want editable/link installation)

## How to use

There are currently two gyms implemented:

### Maze navigation gym

See [docs/maze.md](docs/maze.md).

### Walking robot gym

See [docs/walking_robot.md](docs/walking_robot.md).
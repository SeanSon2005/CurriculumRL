from enum import Enum

#Possible actions
class Action(Enum):
    move_up = 0
    move_down = 1
    move_left = 2
    move_right = 3
    move_up_right = 4
    move_up_left = 5
    move_down_right = 6
    move_down_left = 7
    grab_object = 8
    drop_object = 9
    danger_sensing = 10
    get_messages = 11
    send_message = 12
    wait = 13


# More sub-Tasks to be defined (These are the basic ones)
class Tasks(Enum):
    move = 0
    comms = 1
    grab_object = 2
    drop_object = 3
    
# PsychXR Oculus Rift minimal rendering example. This file is public domain.
#
from psychxr.libovr import *
import sys


def main():

    # create a rift session object
    if failure(initialize()):
        return -1

    if failure(create()):
        shutdown()
        return -1

    hmdDesc = getHmdInfo()

    resolution = hmdDesc.resolution
    print(resolution)  # print the resolution of the display

    destroy()
    shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())


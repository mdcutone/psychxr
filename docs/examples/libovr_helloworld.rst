==================
LibOVR Hello World
==================

A minimal example to create a VR session using the LibOVR interface::

    from psychxr.libovr import *

    # create a rift session object
    if failure(initialize()):
        return -1

    if failure(create()):
        shutdown()
        return -1

    hmdDesc = getHmdInfo()

    resolution = hmdDesc.resolution

    destroy()
    shutdown()


Above is similar to the example shown in "Initialization and Sensor Enumeration"
in the Oculus Rift PC SDK developer guide
(https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-sensor/).
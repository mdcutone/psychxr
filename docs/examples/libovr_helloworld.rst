==================
LibOVR Hello World
==================

A minimal example to create a VR session using the LibOVR interface::

    import psychxr.libovr as libovr

    # create a rift session object
    if libovr.failure(ovr.initialize()):
        return -1

    if libovr.success(ovr.createSession()):
        ovr.shutdown()
        return -1

    resolution = libovr.getScreenSize()

    libovr.destroySession()
    libovr.shutdown()


Above is similar to the example shown in "Initialization and Sensor Enumeration"
in the Oculus Rift PC SDK developer guide
(https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-sensor/).
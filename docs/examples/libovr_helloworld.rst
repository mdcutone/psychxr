===============
Minimal Example
===============

A minimal example to create a VR session using the LibOVR interface::

    import psychxr.libovr as libovr
    import sys


    def main():

        # create a rift session object
        if libovr.failure(libovr.initialize()):
            return -1

        if libovr.failure(libovr.create()):
            libovr.shutdown()
            return -1

        hmdDesc = getHmdInfo()

        resolution = hmdDesc.resolution
        print(resolution)  # print the resolution of HMD raster display

        libovr.destroy()  # clean up
        libovr.shutdown()

        return 0


    if __name__ == "__main__":
        sys.exit(main())


Above is similar to the example shown in `Initialization and Sensor Enumeration
<https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-sensor/>`_
from the Oculus Rift PC SDK developer guide.
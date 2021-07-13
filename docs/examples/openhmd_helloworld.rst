===============
Minimal Example
===============

A minimal example to create a VR session using the OpenHMD interface::

    import sys
    import psychxr.drivers.openhmd as ohmd


    def main():
        # create a session
        if ohmd.failure(ohmd.create()):
            sys.exit(1)  # error, could not create context, exit

        if ohmd.probe() == 0:  # probe for connected devices, returns number found
            print("Cannot find any OpenHMD supported devices on this system!")
            sys.exit(1)

        # get HMD devices
        found_hmds = ohmd.getDevices(ohmd.OHMD_DEVICE_CLASS_HMD)
        if not found_hmds:
            print("Cannot find any HMDs connected to this system!")
            sys.exit(1)

        hmd_device = found_hmds[0]  # use the first HMD found

        # open the device
        ohmd.openDevice(hmd_device)

        # get HMD display info
        res_horiz = ohmd.getDeviceParami(
            hmd_device, ohmd.OHMD_SCREEN_HORIZONTAL_RESOLUTION)
        res_vert = ohmd.getDeviceParami(
            hmd_device, ohmd.OHMD_SCREEN_VERTICAL_RESOLUTION)

        # print the resolution of the display
        print((res_horiz, res_vert))

        # close the device
        ohmd.closeDevice(hmd_device)

        # shutdown the session
        ohmd.destroy()


    if __name__ == "__main__":
        sys.exit(main())

The above example should work without having and HMD connected since OpenHMD
will create a debug (Null) device.

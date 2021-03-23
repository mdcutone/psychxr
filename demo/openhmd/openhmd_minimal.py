# PsychXR OpenHMD minimal example. This file is public domain.
#
# This is similar to the `demo/libovr/libovr_minimal.py` example. Prints the
# resolution of the connected HMD (null device).
#

import sys
import psychxr.drivers.openhmd as ohmd

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
hmd_display_info = ohmd.getDisplayInfo(hmd_device)

# print the resolution of the display
print(hmd_display_info.resolution)

# close the device
ohmd.closeDevice(hmd_device)

# shutdown the session
ohmd.destroy()

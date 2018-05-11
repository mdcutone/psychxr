import psychxr.ovr.capi as capi

color = capi.ovrHmdDesc()
color.DefaultEyeFov[capi.ovrEye_Left].UpTan = 10.0
print(color.DefaultEyeFov[capi.ovrEye_Right].UpTan)
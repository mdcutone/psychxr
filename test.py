import psychxr.ovr.capi as capi

init_pars = capi.ovrInitParams()
init_pars.RequestedMinorVersion = 15

capi.ovr_Initialize(init_pars)
print(capi.ovr_GetVersionString())

session = capi.ovrSession()
print(capi.ovr_Create(session, capi.ovrGraphicsLuid()))
tm = capi.ovr_GetPredictedDisplayTime(session, 0)
tracker_state = capi.ovr_GetTrackingState(session, tm, False)
out_poses = []
pose = capi.ovr_GetDevicePoses(session,
                               [capi.ovrTrackedDevice_HMD],
                               1,
                               tm,
                               out_poses)

print(out_poses)

print(out_poses[0].ThePose.Position.x)

hmd_settings = capi.ovr_GetHmdDesc(session)
print(hmd_settings.Manufacturer)
print(hmd_settings.Resolution.w, hmd_settings.Resolution.h)

capi.ovr_Destroy(session)

capi.ovr_Shutdown()
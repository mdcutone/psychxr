#  =============================================================================
#  libovr_const.pxi - Module level constants
#  =============================================================================
#
#  libovr_const.pxi
#
#  Copyright 2020 Matthew Cutone <cutonem(a)yorku.ca> and Laurie M. Wilcox
#  <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York University,
#  Toronto, Canada
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

# misc constants
EYE_LEFT = capi.ovrEye_Left
EYE_RIGHT = capi.ovrEye_Right
EYE_COUNT = capi.ovrEye_Count
HAND_LEFT = capi.ovrHand_Left
HAND_RIGHT = capi.ovrHand_Right
HAND_COUNT = capi.ovrHand_Count

# performance
MAX_PROVIDED_FRAME_STATS = capi.ovrMaxProvidedFrameStats



# layer header flags
LAYER_FLAG_HIGH_QUALITY = capi.ovrLayerFlag_HighQuality
LAYER_FLAG_TEXTURE_ORIGIN_AT_BOTTOM_LEFT = \
    capi.ovrLayerFlag_TextureOriginAtBottomLeft
LAYER_FLAG_HEAD_LOCKED = capi.ovrLayerFlag_HeadLocked

# HMD types
HMD_NONE = capi.ovrHmd_None
HMD_DK1 = capi.ovrHmd_DK1
HMD_DKHD = capi.ovrHmd_DKHD
HMD_DK2 = capi.ovrHmd_DK2
HMD_CB = capi.ovrHmd_CB
HMD_OTHER = capi.ovrHmd_Other
HMD_E3_2015  = capi.ovrHmd_E3_2015
HMD_ES06 = capi.ovrHmd_ES06
HMD_ES09 = capi.ovrHmd_ES09
HMD_ES11 = capi.ovrHmd_ES11
HMD_CV1 = capi.ovrHmd_CV1
HMD_RIFTS = capi.ovrHmd_RiftS

# logging levels
LOG_LEVEL_DEBUG = capi.ovrLogLevel_Debug
LOG_LEVEL_INFO = capi.ovrLogLevel_Info
LOG_LEVEL_ERROR = capi.ovrLogLevel_Error
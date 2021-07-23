# distutils: language=c++
#  =============================================================================
#  _openxr.pyx - Python Interface Module for OpenXR
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com> and Laurie M.
#  Wilcox <lmwilcox(a)yorku.ca>; The Centre For Vision Research, York
#  University, Toronto, Canada
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
"""This extension module exposes the OpenXR driver interface.
"""

# ------------------------------------------------------------------------------
# Module information
#
__author__ = "Matthew D. Cutone"
__credits__ = []
__copyright__ = "Copyright 2021 Matthew D. Cutone"
__license__ = "MIT"
__version__ = '0.2.4rc2'
__status__ = "Stable"
__maintainer__ = "Matthew D. Cutone"
__email__ = "mcutone@opensciencetools.com"

__all__ = [
    'createInstance',
    'destroyInstance'
]

from . cimport openxr
cimport numpy as np
import numpy as np
np.import_array()
import warnings


# ------------------------------------------------------------------------------
# Module level constants
#

cdef openxr.XrInstance _ptrInstance = NULL  # pointer to instance
cdef openxr.XrSession _ptrSession = NULL  # pointer to session


def createInstance():
    """Create an OpenXR instance."""
    global _ptrInstance

    if _ptrInstance is not NULL:
        return

    cdef openxr.XrGraphicsBindingOpenGLWin32KHR graphics_binding_gl
    cdef openxr.XrApplicationInfo app_info
    app_info.applicationName = ""
    app_info.applicationVersion = 1
    app_info.engineName = ""
    app_info.engineVersion = 0
    app_info.apiVersion = openxr.XR_CURRENT_API_VERSION

    cdef const char* enabled_exts[1]
    enabled_exts[0] = openxr.XR_KHR_OPENGL_ENABLE_EXTENSION_NAME
    cdef openxr.XrInstanceCreateInfo instance_create_info
    instance_create_info.type = openxr.XR_TYPE_INSTANCE_CREATE_INFO
    instance_create_info.next = NULL
    instance_create_info.createFlags = 0
    instance_create_info.enabledExtensionCount = 1
    instance_create_info.enabledExtensionNames = enabled_exts
    instance_create_info.enabledApiLayerCount = 0
    instance_create_info.enabledApiLayerNames = NULL
    instance_create_info.applicationInfo = app_info

    cdef openxr.XrResult result
    result = openxr.xrCreateInstance(&instance_create_info, &_ptrInstance)


def destroyInstance():
    """Destroy an OpenXR instance."""
    pass

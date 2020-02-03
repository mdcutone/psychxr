#  =============================================================================
#  libovr_input.pxi - Input and controller related functions
#  =============================================================================
#
#  libovr_input.pxi
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
# button types
BUTTON_A = capi.ovrButton_A
BUTTON_B = capi.ovrButton_B
BUTTON_RTHUMB = capi.ovrButton_RThumb
BUTTON_RSHOULDER = capi.ovrButton_RShoulder
BUTTON_X = capi.ovrButton_X
BUTTON_Y = capi.ovrButton_Y
BUTTON_LTHUMB = capi.ovrButton_LThumb
BUTTON_LSHOULDER = capi.ovrButton_LShoulder
BUTTON_UP = capi.ovrButton_Up
BUTTON_DOWN = capi.ovrButton_Down
BUTTON_LEFT = capi.ovrButton_Left
BUTTON_RIGHT = capi.ovrButton_Right
BUTTON_ENTER = capi.ovrButton_Enter
BUTTON_BACK = capi.ovrButton_Back
BUTTON_VOLUP = capi.ovrButton_VolUp
BUTTON_VOLDOWN = capi.ovrButton_VolDown
BUTTON_HOME = capi.ovrButton_Home
BUTTON_PRIVATE = capi.ovrButton_Private
BUTTON_RMASK = capi.ovrButton_RMask
BUTTON_LMASK = capi.ovrButton_LMask

# touch types
TOUCH_A = capi.ovrTouch_A
TOUCH_B = capi.ovrTouch_B
TOUCH_RTHUMB = capi.ovrTouch_RThumb
TOUCH_RTHUMBREST = capi.ovrTouch_RThumbRest
TOUCH_X = capi.ovrTouch_X
TOUCH_Y = capi.ovrTouch_Y
TOUCH_LTHUMB = capi.ovrTouch_LThumb
TOUCH_LTHUMBREST = capi.ovrTouch_LThumbRest
TOUCH_LINDEXTRIGGER = capi.ovrTouch_LIndexTrigger
TOUCH_RINDEXPOINTING = capi.ovrTouch_RIndexPointing
TOUCH_RTHUMBUP = capi.ovrTouch_RThumbUp
TOUCH_LINDEXPOINTING = capi.ovrTouch_LIndexPointing
TOUCH_LTHUMBUP = capi.ovrTouch_LThumbUp

# controller types
CONTROLLER_TYPE_NONE = capi.ovrControllerType_None
CONTROLLER_TYPE_XBOX = capi.ovrControllerType_XBox
CONTROLLER_TYPE_REMOTE = capi.ovrControllerType_Remote
CONTROLLER_TYPE_TOUCH = capi.ovrControllerType_Touch
CONTROLLER_TYPE_LTOUCH = capi.ovrControllerType_LTouch
CONTROLLER_TYPE_RTOUCH = capi.ovrControllerType_RTouch
CONTROLLER_TYPE_OBJECT0 = capi.ovrControllerType_Object0
CONTROLLER_TYPE_OBJECT1 = capi.ovrControllerType_Object1
CONTROLLER_TYPE_OBJECT2 = capi.ovrControllerType_Object2
CONTROLLER_TYPE_OBJECT3 = capi.ovrControllerType_Object3

# controller states
cdef capi.ovrInputState[9] _inputStates
cdef capi.ovrInputState[9] _prevInputState


def getConnectedControllerTypes():
    """Get connected controller types.

    Returns
    -------
    list of int
        IDs of connected controller types. Possible values returned are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    See Also
    --------
    updateInputState : Poll a controller's current state.

    Examples
    --------

    Check if the left touch controller is paired::

        controllers = getConnectedControllerTypes()
        hasLeftTouch = CONTROLLER_TYPE_LTOUCH in controllers

    Update all connected controller states::

        for controller in getConnectedControllerTypes():
            result, time = updateInputState(controller)

    """
    global _ptrSession
    cdef unsigned int result = capi.ovr_GetConnectedControllerTypes(
        _ptrSession)

    cdef list toReturn = list()
    if (capi.ovrControllerType_XBox & result) == capi.ovrControllerType_XBox:
        toReturn.append(CONTROLLER_TYPE_XBOX)
    if (capi.ovrControllerType_Remote & result) == capi.ovrControllerType_Remote:
        toReturn.append(CONTROLLER_TYPE_REMOTE)
    if (capi.ovrControllerType_Touch & result) == capi.ovrControllerType_Touch:
        toReturn.append(CONTROLLER_TYPE_TOUCH)
    if (capi.ovrControllerType_LTouch & result) == capi.ovrControllerType_LTouch:
        toReturn.append(CONTROLLER_TYPE_LTOUCH)
    if (capi.ovrControllerType_RTouch & result) == capi.ovrControllerType_RTouch:
        toReturn.append(CONTROLLER_TYPE_RTOUCH)
    if (capi.ovrControllerType_Object0 & result) == capi.ovrControllerType_Object0:
        toReturn.append(CONTROLLER_TYPE_OBJECT0)
    if (capi.ovrControllerType_Object1 & result) == capi.ovrControllerType_Object1:
        toReturn.append(CONTROLLER_TYPE_OBJECT1)
    if (capi.ovrControllerType_Object2 & result) == capi.ovrControllerType_Object2:
        toReturn.append(CONTROLLER_TYPE_OBJECT2)
    if (capi.ovrControllerType_Object3 & result) == capi.ovrControllerType_Object3:
        toReturn.append(CONTROLLER_TYPE_OBJECT3)

    return toReturn


def updateInputState(int controller):
    """Refresh the input state of a controller.

    Subsequent :func:`getButton`, :func:`getTouch`, :func:`getThumbstickValues`,
    :func:`getIndexTriggerValues`, and :func:`getHandTriggerValues` calls using
    the same `controller` value will reflect the new state.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple (int, float)
        Result of the ``OVR::ovr_GetInputState`` LibOVR API call and polling
        time in seconds.

    See Also
    --------
    getConnectedControllerTypes : Get a list of connected controllers.
    getButton: Get button states.
    getTouch: Get touches.

    """
    global _prevInputState
    global _inputStates
    global _ptrSession

    # get the controller index in the states array
    cdef int idx
    cdef capi.ovrInputState* previousInputState
    cdef capi.ovrInputState* currentInputState
    cdef capi.ovrResult result

    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    previousInputState = &_prevInputState[idx]
    currentInputState = &_inputStates[idx]

    # copy the current input state into the previous before updating
    previousInputState[0] = currentInputState[0]

    # get the current input state
    result = capi.ovr_GetInputState(
        _ptrSession,
        <capi.ovrControllerType>controller,
        currentInputState)

    return result, currentInputState.TimeInSeconds


def getButton(int controller, int button, str testState='continuous'):
    """Get a button state.

    The `controller` to test is specified by its ID, defined as constants
    starting with :data:`CONTROLLER_TYPE_*`. Buttons to test are
    specified using their ID, defined as constants starting with
    :data:`BUTTON_*`. Button IDs can be ORed together for testing
    multiple button states. The returned value represents the button state
    during the last :func:`updateInputState` call for the specified
    `controller`.

    An optional trigger mode may be specified which defines the button's
    activation criteria. By default, `testState` is 'continuous' will return the
    immediate state of the button. Using 'rising' (or 'pressed') will
    return True once when the button transitions to being pressed between
    subsequent :func:`updateInputState` calls, whereas 'falling' (and
    'released') will return True once the button is released. If
    :func:`updateInputState` was called only once, 'rising' and 'falling' will
    return False.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    button : int
        Button to check. Values can be ORed together to test for multiple button
        presses. If a given controller does not have a particular button, False
        will always be returned. Valid button values are:

        * ``BUTTON_A``
        * ``BUTTON_B``
        * ``BUTTON_RTHUMB``
        * ``BUTTON_RSHOULDER``
        * ``BUTTON_X``
        * ``BUTTON_Y``
        * ``BUTTON_LTHUMB``
        * ``BUTTON_LSHOULDER``
        * ``BUTTON_UP``
        * ``BUTTON_DOWN``
        * ``BUTTON_LEFT``
        * ``BUTTON_RIGHT``
        * ``BUTTON_ENTER``
        * ``BUTTON_BACK``
        * ``BUTTON_VOLUP``
        * ``BUTTON_VOLDOWN``
        * ``BUTTON_HOME``
        * ``BUTTON_PRIVATE``
        * ``BUTTON_RMASK``
        * ``BUTTON_LMASK``

    testState : str
        State to test buttons for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple (bool, float)
        Result of the button press and the time in seconds it was polled.

    See Also
    --------
    getTouch : Get touches.

    Examples
    --------
    Check if the 'X' button on the touch controllers was pressed::

        isPressed = getButtons(CONTROLLER_TYPE_TOUCH,
            BUTTON_X, 'pressed')

    Test for multiple buttons (e.g. 'X' and 'Y') being released::

        buttons = BUTTON_X | BUTTON_Y
        controller = CONTROLLER_TYPE_TOUCH
        isReleased = getButtons(controller, buttons, 'released')

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # get the time the controller was polled
    cdef double t_sec = _inputStates[idx].TimeInSeconds

    # pointer to the current and previous input state
    cdef unsigned int curButtons = _inputStates[idx].Buttons
    cdef unsigned int prvButtons = _prevInputState[idx].Buttons

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curButtons & button) == button
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curButtons & button) == button and \
                      (prvButtons & button) != button
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curButtons & button) != button and \
                      (prvButtons & button) == button
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult, t_sec


def getTouch(int controller, int touch, str testState='continuous'):
    """Get a touch state.

    The `controller` to test is specified by its ID, defined as constants
    starting with :data:`CONTROLLER_TYPE_*`. Touches to test are
    specified using their ID, defined as constants starting with
    :data:`TOUCH_*`. Touch IDs can be ORed together for testing multiple
    touch states. The returned value represents the touch state during the last
    :func:`updateInputState` call for the specified `controller`.

    An optional trigger mode may be specified which defines a touch's
    activation criteria. By default, `testState` is 'continuous' will return the
    immediate state of the button. Using 'rising' (or 'pressed') will
    return ``True`` once when something is touched between subsequent
    :func:`updateInputState` calls, whereas 'falling' (and 'released') will
    return ``True`` once the touch is discontinued. If :func:`updateInputState`
    was called only once, 'rising' and 'falling' will return False.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

        However, touches are only applicable for devices which support that
        feature.

    touch : int
        Touch to check. Values can be ORed together to test for multiple
        touches. If a given controller does not have a particular touch,
        ``False`` will always be returned. Valid touch values are:

        * ``TOUCH_A``
        * ``TOUCH_B``
        * ``TOUCH_RTHUMB``
        * ``TOUCH_RSHOULDER``
        * ``TOUCH_X``
        * ``TOUCH_Y``
        * ``TOUCH_LTHUMB``
        * ``TOUCH_LSHOULDER``
        * ``TOUCH_LINDEXTRIGGER``
        * ``TOUCH_LINDEXTRIGGER``
        * ``TOUCH_LTHUMBREST``
        * ``TOUCH_RTHUMBREST``
        * ``TOUCH_RINDEXPOINTING``
        * ``TOUCH_RTHUMBUP``
        * ``TOUCH_LINDEXPOINTING``
        * ``TOUCH_LTHUMBUP``

    testState : str
        State to test touches for. Valid states are 'rising', 'falling',
        'continuous', 'pressed', and 'released'.

    Returns
    -------
    tuple (bool, float)
        Result of the touches and the time in seconds it was polled.

    See Also
    --------
    getButton : Get a button state.

    Notes
    -----
    * Special 'touches' ``TOUCH_RINDEXPOINTING``, ``TOUCH_RTHUMBUP``,
      ``TOUCH_RTHUMBREST``, ``TOUCH_LINDEXPOINTING``, ``TOUCH_LINDEXPOINTING``,
      and ``TOUCH_LINDEXPOINTING``, can be used to recognise hand pose/gestures.

    Examples
    --------
    Check if the user is making a pointing gesture with their right index
    finger::

        isPointing = getTouch(
            controller=CONTROLLER_TYPE_LTOUCH,
            touch=TOUCH_LINDEXPOINTING)

    """
    global _prevInputState
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # get the time the controller was polled
    cdef double t_sec = _inputStates[idx].TimeInSeconds

    # pointer to the current and previous input state
    cdef unsigned int curTouches = _inputStates[idx].Touches
    cdef unsigned int prvTouches = _prevInputState[idx].Touches

    # test if the button was pressed
    cdef bint stateResult = False
    if testState == 'continuous':
        stateResult = (curTouches & touch) == touch
    elif testState == 'rising' or testState == 'pressed':
        # rising edge, will trigger once when pressed
        stateResult = (curTouches & touch) == touch and \
                      (prvTouches & touch) != touch
    elif testState == 'falling' or testState == 'released':
        # falling edge, will trigger once when released
        stateResult = (curTouches & touch) != touch and \
                      (prvTouches & touch) == touch
    else:
        raise ValueError("Invalid trigger mode specified.")

    return stateResult, t_sec


def getThumbstickValues(int controller, bint deadzone=False):
    """Get analog thumbstick values.

    Get the values indicating the displacement of the controller's analog
    thumbsticks. Returns two tuples for the up-down and left-right of each
    stick. Values range from -1 to 1.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    deadzone : bool
        Apply a deadzone if True.

    Returns
    -------
    tuple (float, float)
        Thumbstick values.

    Examples
    --------

    Get the thumbstick values with deadzone for the touch controllers::

        ovr.updateInputState()  # get most recent input state
        leftThumbStick, rightThumbStick = ovr.getThumbstickValues(
            ovr.CONTROLLER_TYPE_TOUCH, deadzone=True)
        x, y = rightThumbStick  # left-right, up-down values for right stick

    """
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float thumbstick_x0 = 0.0
    cdef float thumbstick_y0 = 0.0
    cdef float thumbstick_x1 = 0.0
    cdef float thumbstick_y1 = 0.0

    if deadzone:
        thumbstick_x0 = currentInputState[0].Thumbstick[0].x
        thumbstick_y0 = currentInputState[0].Thumbstick[0].y
        thumbstick_x1 = currentInputState[0].Thumbstick[1].x
        thumbstick_y1 = currentInputState[0].Thumbstick[1].y
    else:
        thumbstick_x0 = currentInputState[0].ThumbstickNoDeadzone[0].x
        thumbstick_y0 = currentInputState[0].ThumbstickNoDeadzone[0].y
        thumbstick_x1 = currentInputState[0].ThumbstickNoDeadzone[1].x
        thumbstick_y1 = currentInputState[0].ThumbstickNoDeadzone[1].y

    return np.array((thumbstick_x0, thumbstick_y0), dtype=np.float32), \
           np.array((thumbstick_x1, thumbstick_y1), dtype=np.float32)


def getIndexTriggerValues(int controller, bint deadzone=False):
    """Get analog index trigger values.

    Get the values indicating the displacement of the controller's analog
    index triggers. Returns values for the left an right sticks. Values range
    from -1 to 1.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple (float, float)
        Trigger values (left, right).

    See Also
    --------
    getThumbstickValues : Get thumbstick displacements.
    getHandTriggerValues : Get hand trigger values.

    Examples
    --------

    Get the index trigger values for touch controllers (with deadzone)::

        leftVal, rightVal = getIndexTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

    Cast a ray from the controller when a trigger is pulled::

        _, rightVal = getIndexTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

        # handPose of right hand from the last tracking state
        if rightVal > 0.75:  # 75% thresholds
            if handPose.raycastSphere(target):  # target is LibOVRPose
                print('Target hit!')
            else:
                print('Missed!')

    """
    # convert the string to an index
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float triggerLeft = 0.0
    cdef float triggerRight = 0.0

    if deadzone:
        triggerLeft = currentInputState[0].IndexTrigger[0]
        triggerRight = currentInputState[0].IndexTrigger[1]
    else:
        triggerLeft = currentInputState[0].IndexTriggerNoDeadzone[0]
        triggerRight = currentInputState[0].IndexTriggerNoDeadzone[1]

    return np.array((triggerLeft, triggerRight), dtype=np.float32)


def getHandTriggerValues(int controller, bint deadzone=False):
    """Get analog hand trigger values.

    Get the values indicating the displacement of the controller's analog
    hand triggers. Returns two values for the left and right sticks. Values
    range from -1 to 1.

    Parameters
    ----------
    controller : int
        Controller name. Valid values are:

        * ``CONTROLLER_TYPE_XBOX`` : XBox gamepad.
        * ``CONTROLLER_TYPE_REMOTE`` : Oculus Remote.
        * ``CONTROLLER_TYPE_TOUCH`` : Combined Touch controllers.
        * ``CONTROLLER_TYPE_LTOUCH`` : Left Touch controller.
        * ``CONTROLLER_TYPE_RTOUCH`` : Right Touch controller.
        * ``CONTROLLER_TYPE_OBJECT0`` : Object 0 controller.
        * ``CONTROLLER_TYPE_OBJECT1`` : Object 1 controller.
        * ``CONTROLLER_TYPE_OBJECT2`` : Object 2 controller.
        * ``CONTROLLER_TYPE_OBJECT3`` : Object 3 controller.

    Returns
    -------
    tuple (float, float)
        Trigger values (left, right).

    See Also
    --------
    getThumbstickValues : Get thumbstick displacements.
    getIndexTriggerValues : Get index trigger values.

    Examples
    --------

    Get the hand trigger values for touch controllers (with deadzone)::

        leftVal, rightVal = getHandTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

    Grip an object if near a hand. Simply set the pose of the object to match
    that of the hand when gripping within some distance of the object's
    origin. When the grip is released, the object will assume the last pose
    before being released. Here is a very basic example of object gripping::

        _, rightVal = getHandTriggerValues(CONTROLLER_TYPE_TOUCH,
            deadzone=True)

        # thing and handPose are LibOVRPoses, handPose is from tracking state
        distanceToHand = abs(handPose.distanceTo(thing.pos))
        if rightVal > 0.75 and distanceToHand < 0.01:
            thing.posOri = handPose.posOri

    """
    global _inputStates

    # get the controller index in the states array
    cdef int idx
    if controller == CONTROLLER_TYPE_XBOX:
        idx = 0
    elif controller == CONTROLLER_TYPE_REMOTE:
        idx = 1
    elif controller == CONTROLLER_TYPE_TOUCH:
        idx = 2
    elif controller == CONTROLLER_TYPE_LTOUCH:
        idx = 3
    elif controller == CONTROLLER_TYPE_RTOUCH:
        idx = 4
    elif controller == CONTROLLER_TYPE_OBJECT0:
        idx = 5
    elif controller == CONTROLLER_TYPE_OBJECT1:
        idx = 6
    elif controller == CONTROLLER_TYPE_OBJECT2:
        idx = 7
    elif controller == CONTROLLER_TYPE_OBJECT3:
        idx = 8
    else:
        raise ValueError("Invalid controller type specified.")

    # pointer to the current and previous input state
    cdef capi.ovrInputState* currentInputState = &_inputStates[idx]

    cdef float triggerLeft = 0.0
    cdef float triggerRight = 0.0

    if deadzone:
        triggerLeft = currentInputState[0].HandTrigger[0]
        triggerRight = currentInputState[0].HandTrigger[1]
    else:
        triggerLeft = currentInputState[0].HandTriggerNoDeadzone[0]
        triggerRight = currentInputState[0].HandTriggerNoDeadzone[1]

    return np.array((triggerLeft, triggerRight), dtype=np.float32)

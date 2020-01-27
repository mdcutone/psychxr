#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Placeholder module for the `libovr` driver interface."""

import warnings

warnings.warn("Module `psychxr.libovr` has moved to `psychxr.drivers.libovr`. "
              "This module will be removed in future versions of PsychXR.",
              DeprecationWarning)

from psychxr.drivers.libovr import *

Colorimetry Data for Oculus HMDs
================================

*Published by Matthew Cutone on 14/03/2021*

The version 23.0 release of the Oculus PC SDK exposes a color management API
which has been made accessible through *PsychXR*. The purpose of the API seems
to ensure that content authored to one display appears "correct" on others if
color gamuts vary between them (due to different display technologies in use).
Going forward, this is a feature that users of *PsychXR* should be aware of and
consider using, since perceived colors may vary greatly across displays if not
accounted for.

However, the use of this API is not the focus of this article. The new API
documentation provides lots of colorimetry information about various HMDs in the
Oculus product lineup. Such data may be of interest to researchers and users, so
I present them here for reference.

Chromaticity Coordinates
------------------------

The device manufacturer lists the *nominal* chromaticity coordinates (CIE 1931
xy) of RGB primaries and white points for their displays [1]_. The values are
represented in the following table:

============== ================== ================== ================== ==================
HMD Model                Red (xy)         Green (xy)          Blue (xy)   White Point (xy)
============== ================== ================== ================== ==================
Rift CV1       (0.666, 0.334)     (0.238, 0.714)     (0.139, 0.053)     (0.298, 0.318)
Rift S         (0.640, 0.330)     (0.292, 0.586)     (0.156, 0.058)     (0.156, 0.058)
Quest          (0.661, 0.338)     (0.228, 0.718)     (0.142, 0.042)     (0.298, 0.318)
============== ================== ================== ================== ==================

Below is a CIE 1931 xy chromaticity diagram showing the color models of each HMD
based on the provided data. The ITU-R BT.709 color model [2]_ is also shown for
comparison.

.. image:: ../_static/hmd_chroma_diagram.png
  :alt: CIE 1931 xy chromaticity diagram showing the gamuts of various displays.

We see that the Quest and Rift CV1 have a considerably larger color gamut than
the Rift S. This difference seems to have necessitated the addition of the color
management API, as image content authored to the CV1 and Quest platforms will
appear less saturated when presented on the Rift S unless colors are remapped.

References
----------

.. [1] From the ``OVR_API.h`` file in the Oculus PC SDK version 23.0 source
       code, retrieved 2012-03-14
.. [2] `Rec. 709 <https://en.wikipedia.org/wiki/Rec._709>`_ from Wikipedia, the
       free encyclopedia. Retrieved 2021-03-15.
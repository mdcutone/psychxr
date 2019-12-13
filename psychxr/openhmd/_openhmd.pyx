# distutils: language=c++
__all__ = ['create', 'destroy']

# --------------------------------------
# Initialize module
#
cdef ohmd_context* _ctx = NULL


def create():
    """Create a new OpenHMD context/session."""
    global _ctx
    _ctx = ohmd_ctx_create()

    if _ctx is not NULL:
        return 0

    return 1


def destroy():
    """Destroy the current context/session."""
    global _ctx
    ohmd_ctx_destroy(_ctx)
    _ctx = NULL
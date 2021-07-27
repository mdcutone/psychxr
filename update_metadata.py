#  =============================================================================
#  update_metadata.py - Update module metadata project-wide
#  =============================================================================
#
#  Copyright 2021 Matthew Cutone <mcutone@opensciencetools.com>
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
"""Script to update module metadata information across PsychXR source files.
This script should only be used when packaging new releases.

"""

import os

# Set this to the project version of the current release. Development branches
# should use suffix ".devN".
PSYCHXR_VERSION_STRING = '0.2.4rc3.post0'

# file extensions which contain source code in the project
SRC_EXTENSIONS = ['.py', '.pyx', '.pxd']


def replace_version_in_file(file_path, version_string):
    """Replace version information in the file's header with the indicated
    value.

    Parameters
    ----------
    file_path : str
        Path to file to update version information.
    version_string : str
        Version string (e.g., "0.2.4rc1", etc.)

    """
    with open(file_path, 'r') as f:
        src_lines = f.readlines()

    # find the line which has the version string
    for i, line_text in enumerate(src_lines):
        if line_text.startswith("__version__"):
            src_lines[i] = "__version__ = '{}'\n".format(version_string)

    with open(file_path, 'w') as f:
        f.writelines(src_lines)


def update_version_in_project_files(version_string):
    """Update version information in all project files.

    Parameters
    ----------
    version_string : str
        Version string (e.g., "0.2.4rc1", etc.)

    """
    for dir_name, dirs, files in os.walk("."):
        for file in files:
            if not any([file.endswith(ext) for ext in SRC_EXTENSIONS]):
                continue
            file_path = os.path.join(dir_name, file)
            replace_version_in_file(file_path, version_string)


# Usually this is run for the setup.py script
if __name__ == "__main__":
    update_version_in_project_files(PSYCHXR_VERSION_STRING)

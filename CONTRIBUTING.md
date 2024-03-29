# Contributing

When contributing to this repository, please first discuss the change you wish 
to make via issue, email, or any other method with the owners of this repository 
before making a change.

Please note we have a code of conduct, please follow it in all your interactions 
with the project.

## Pull Request Process

1. Submit a pull request targeting the `main` branch, describing in detail what 
   you are merging and why.
2. Once the review process is complete, the project maintainer will choose pull 
   in the changes or reject them.
   
## What's needed?

Small bug fixes and documentation corrections are always welcome, however you 
should still submit an issue for more substantial changes beforehand. 

For more substantial changes (e.g., adding a new driver interface, refactoring,
etc.) it is advised that you submit an issue outlining the proposed changes
prior to starting any work on your PR. 

Help with implementing new driver interfaces for HMDs (e.g., OpenXR, OpenHMD, 
etc.) and other VR related devices is most appreciated. Driver interfaces 
require some level of standardization, were they should share common design and 
usage patterns where practical. All driver interface extensions included with 
PsychXR **MUST** be written in *Cython*. Any interface module using *CFFI* or 
*ctypes* based wrappers will not be accepted.

Contributors who have made substantial contributions to the project may be added 
to the list of authors and acknowledged in future publications related to 
PsychXR.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of 
experience, nationality, personal appearance, race, religion, or sexual identity 
and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team at matthew.cutone(at)gmail.com. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an 
incident. Further details of specific enforcement policies may be posted 
separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], 
version 1.4, available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/
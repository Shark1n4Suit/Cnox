# SAIC Project TODO

This document tracks the development roadmap and planned features for the Secure AI Container (SAIC) project.

## Future Backend Support

Our hardware expansion plan to support diverse AI inference accelerators:

- [ ] Intel Arc GPU: Enhance the existing OpenVINO backend to allow device selection (CPU or GPU).

- [ ] AMD Hardware (CPU & GPU): Create a new AmdBackend by integrating AMD's ROCm/MIGraphX SDK.

- [ ] Apple Silicon (M1/M2/M3): Create a new AppleBackend by integrating Apple's Core ML framework.

- [ ] Research: Investigate creating a cross-platform DirectMlBackend for Windows using Microsoft's DirectML API. 
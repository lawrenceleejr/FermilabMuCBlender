# Headless Blender + the host-side Python the tools need, for reproducible
# renders in CI or on any machine without a local Blender.
#
# GPU rendering is done on the host through ./render.sh --gpu; this image is
# CPU-only on purpose. It exists so a CI render and a local render are the same
# render, which matters here because the scene is assembled from data and a
# silent difference in a dependency would show up as a difference in geometry.
FROM debian:bookworm-slim

ARG BLENDER_SERIES=5.2
ARG BLENDER_VERSION=5.2.1
ARG BLENDER_URL=https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz

# Libraries Blender needs even in background mode. libegl1 is not optional
# despite --background: without it Blender exits on "libEGL.so.1: cannot open
# shared object file" before it reaches the script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils ffmpeg \
        python3 python3-pip python3-numpy python3-pil python3-matplotlib \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
        libgl1 libglu1-mesa libegl1 libsm6 libxext6 libxrandr2 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fSL "$BLENDER_URL" -o /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/blender /usr/local/bin/blender

RUN blender --version && python3 -c "import numpy, PIL, matplotlib; print('host python ok')"

WORKDIR /work
ENTRYPOINT ["/bin/bash"]

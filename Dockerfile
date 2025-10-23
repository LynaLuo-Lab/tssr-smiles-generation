FROM mambaorg/micromamba:1.5.8
SHELL ["/bin/bash","-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /workspace

# env layer
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean --all -y

# project
COPY . /workspace

# ⬇️ add these three lines
USER root
COPY --chown=$MAMBA_USER:$MAMBA_USER --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
USER $MAMBA_USER

ENTRYPOINT ["micromamba", "run", "-n", "base", "/usr/local/bin/entrypoint.sh"]
CMD ["help"]

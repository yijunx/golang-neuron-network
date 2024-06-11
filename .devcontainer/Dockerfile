FROM ubuntu:latest

ENV GO_VERSION=1.21.5
ENV CGO_ENABLED=1
ENV GOOS=linux
ENV GOARCH=arm64

ARG DOCKER_HOME="/opt/yijun"
ARG DOCKER_CODE="/opt/yijun/code"
ARG DOCKER_GROUP="yijun"
ARG DOCKER_USER="yijun"
ARG DOCKER_UID=5000

WORKDIR ${DOCKER_CODE}

RUN groupadd -g ${DOCKER_UID} ${DOCKER_GROUP} \
    && useradd -r -u ${DOCKER_UID} -g ${DOCKER_GROUP} -d ${DOCKER_HOME} ${DOCKER_USER} \
    && chown -R ${DOCKER_USER}:${DOCKER_GROUP} ${DOCKER_HOME}

RUN apt-get update && apt-get install -y wget git gcc 

RUN wget -P /tmp "https://dl.google.com/go/go${GO_VERSION}.linux-${GOARCH}.tar.gz"
RUN tar -C /usr/local -xzf "/tmp/go${GO_VERSION}.linux-${GOARCH}.tar.gz"
RUN rm "/tmp/go${GO_VERSION}.linux-${GOARCH}.tar.gz"

# ENV GOPATH /opt/go/code
ENV PATH /usr/local/go/bin:$PATH
# # here we add the code folder into path, then we dont need go.mod anymore
# RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

USER ${DOCKER_USER}

ENTRYPOINT [ "/bin/sh", "-c" ]
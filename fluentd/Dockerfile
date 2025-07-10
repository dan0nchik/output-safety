FROM fluent/fluentd:v1.15-alpine
USER root

# Install build dependencies for native gem compilation
RUN apk update \
 && apk add --no-cache --virtual .build-deps \
    build-base \
    ruby-dev \
    openssl-dev \
 && gem install fluent-plugin-mongo fluent-plugin-gelf \
 && apk del .build-deps

USER fluent
COPY fluent.conf /fluentd/etc/fluent.conf

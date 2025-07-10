FROM fluent/fluentd:v1.16-debian

USER root

RUN apt-get update \
 && apt-get install -y build-essential ruby-dev libssl-dev \
 && gem install fluent-plugin-mongo fluent-plugin-gelf \
 && apt-get purge -y build-essential ruby-dev libssl-dev \
 && rm -rf /var/lib/apt/lists/*

USER fluent
COPY fluent.conf /fluentd/etc/fluent.conf

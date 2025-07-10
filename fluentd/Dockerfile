FROM fluent/fluentd:v1.16-debian

USER root

RUN gem install fluent-plugin-kafka fluent-plugin-gelf

USER fluent
COPY fluent.conf /fluentd/etc/fluent.conf
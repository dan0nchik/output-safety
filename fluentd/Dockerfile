FROM fluent/fluentd:v1.15
USER root
RUN gem install fluent-plugin-mongo fluent-plugin-gelf
COPY fluent.conf /fluentd/etc/fluent.conf

FROM apache/spark:latest

USER root

RUN mkdir -p /opt/spark/logs && chmod a+wr /opt/spark/logs

USER 185

ENV SPARK_NO_DAEMONIZE=true

CMD ["/opt/spark/sbin/start-master.sh"]
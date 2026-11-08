# Airflow image with Java (for PySpark) and the Python libraries the pipeline needs
FROM apache/airflow:2.10.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 (headless) for Spark; procps for 'ps'; bash as the shell
RUN rm -f /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    ln -s /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-17-openjdk

# the JDK folder name depends on the CPU (amd64 on most PCs, arm64 on Apple Silicon),
# so it is linked to one fixed path above
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app
COPY requirements.txt ./

USER airflow
RUN pip install --no-cache-dir -r requirements.txt

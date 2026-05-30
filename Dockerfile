FROM jupyter/pyspark-notebook:spark-3.5.0

# Switch to root to install system packages
USER root

# Install Java 17 and replace any existing Java
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Force Java 17 as the active JVM
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Switch back to notebook user
USER ${NB_UID}

# Install Python dependencies
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    numpy

# Expose Jupyter and Spark UI ports
EXPOSE 8888 4040

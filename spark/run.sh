#!/bin/bash

set -e

sbt package
/usr/share/apache-spark/bin/spark-submit \
    --class "Solution" \
    --master 'local[4]' \
    --driver-class-path /usr/share/java/sqlite-jdbc/sqlite-jdbc.jar \
    --jars /usr/share/java/sqlite-jdbc/sqlite-jdbc.jar \
    target/scala-2.11/solution_2.11-1.0.jar \


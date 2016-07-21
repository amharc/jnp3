name := "Solution"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "1.6.0",
    "org.apache.spark" %% "spark-sql" % "1.6.0",
    "org.apache.spark" %% "spark-mllib" % "1.6.0",
    "org.xerial" % "sqlite-jdbc" % "3.8.6"
)

name := "RandLA"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.2.0"

libraryDependencies +=  "org.scalatest" % "scalatest_2.10.0-M4" % "1.9-2.10.0-M4-B1"

libraryDependencies +=  "junit" % "junit" % "4.8.1" % "test"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.2.0"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.2.0"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "1.2.0"


libraryDependencies += "commons-cli" % "commons-cli" % "1.2"

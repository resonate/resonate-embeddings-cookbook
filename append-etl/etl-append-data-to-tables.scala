// start your spark shell

// This script is designed to get your team started working with data append
// You may set the paths and paste it into spark shell via paste mode, :paste followed by ctrl-D to run
// Or bake it into a script and run it via spark-submit
// The script reads in the data append and taxonomy, processes the data append, and saves the results as parquet files
// The schemas are essentially ID, attribute, and values are the selected value for the attribute
// Single selects are things like "which one of the the following options" - so the values are the selected option
// Single selects may still have null values if we assess the question does not apply to the record
// Multi selects are "yes or no" conditions, so the values are 0 or 1

import org.apache.spark.sql._
import org.apache.spark.sql.functions._

// Define parameters for data paths and output directory
case class Params(
                   dataAppendPath: String = "path to your dataAppend file (csv.gz)",
                   dataAppendTaxonomyPath: String = "path to your data append taxonomy (csv)",
                   outDir: String = "where you want to save your files, as parquet"
                 )

val params = Params()

// Define UDFs

// UDF to split the 'ATTRIBUTES' column into an array of svkeys
val splitSvkeys = udf((attributes: String) => attributes.split(",\\s*"))


// Read in the data assets
val dataAppend = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(params.dataAppendPath)

val dataAppendTaxonomy = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(params.dataAppendTaxonomyPath)

// Process 'dataAppend' by splitting 'ATTRIBUTES' into 'trueSvkeys' and cache the result
val dataAppendWithSvkeys = dataAppend
  .withColumn("trueSvkeys", splitSvkeys(col("ATTRIBUTES")))
  .cache()

// Trigger an action to cache the DataFrame
dataAppendWithSvkeys.count()

// Narrow the taxonomic scope by extracting distinct svkeys
val superSetSvkeys = dataAppendWithSvkeys
  .select(explode(col("trueSvkeys")).as("svkey"))
  .distinct()
  .as[String]
  .collect()

// Filter taxonomy based on the relevant svkeys and create 'attributeString'
val focalTaxonomy = dataAppendTaxonomy
  .where(col("survey_value_key").isin(superSetSvkeys: _*))
  .withColumn("attributeString", concat_ws("_", col("AttributeKey"), col("survey_value_key")))

// Create a map for single select attributes
val singleSelectColumnMap: Map[String, String] = focalTaxonomy
  .filter(col("Attribute Type") === "Single Select")
  .select("survey_value_key", "AttributeKey")
  .distinct()
  .as[(Int, String)]
  .collect()
  .map { case (svkey, attrKey) => svkey.toString -> attrKey.toString }
  .toMap


// Extract distinct single and multi-select attributes
val singleSelectAttributes = singleSelectColumnMap.values.toArray.distinct
val multiSelectAttributes = focalTaxonomy
  .filter(col("Attribute Type") === "Multi Select")
  .select("attributeString")
  .distinct()
  .as[String]
  .collect()

val allAttributes = focalTaxonomy
  .select("attributeString")
  .distinct()
  .as[String]
  .collect()

// Broadcast the singleSelectColumnMap for efficient lookup in UDF
val broadcastMap = spark.sparkContext.broadcast(singleSelectColumnMap)
// UDF to map svkeys to a single select attribute
val mapSvkeyToAttr = udf((trueSvkeys: Seq[String], attr: String) => {
  // Access the broadcasted map within the UDF's closure
  trueSvkeys
    .flatMap(svkey => broadcastMap.value.get(svkey))
    .find(_ == attr)
    .getOrElse(null)
})

// Create DataFrame for single selects by adding a new column for each single select attribute
val singleSelectDF = singleSelectAttributes.foldLeft(dataAppendWithSvkeys) { (df, attr) =>
  df.withColumn(
    attr,
    mapSvkeyToAttr(col("trueSvkeys"), lit(attr))
  )
}.drop("trueSvkeys", "ATTRIBUTES")

// Process multi-select attributes
val multiSelectDF = dataAppendWithSvkeys
  .select(col("ID"), explode(col("trueSvkeys")).as("survey_value_key"))
  .join(focalTaxonomy, "survey_value_key")
  .groupBy("ID")
  .pivot("attributeString", multiSelectAttributes)
  .agg(count("survey_value_key"))
  .na.fill(0)

// Alternatively, process all attributes as multi-selects
val allAsMultiSelectDF = dataAppendWithSvkeys
  .select(col("ID"), explode(col("trueSvkeys")).as("survey_value_key"))
  .join(focalTaxonomy, "survey_value_key")
  .groupBy("ID")
  .pivot("attributeString", allAttributes)
  .agg(count("survey_value_key"))
  .na.fill(0)

// Save the processed DataFrames
singleSelectDF.write.mode("overwrite").save(params.outDir + "singleSelectDF")
multiSelectDF.write.mode("overwrite").save(params.outDir + "multiSelectDF")
allAsMultiSelectDF.write.mode("overwrite").save(params.outDir + "allAsMultiSelectDF")
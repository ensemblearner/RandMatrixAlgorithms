package randla
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrices, Vectors}

object SparkApp {

  final def sample[A](dist: Map[A, Double]): A = {
    val p = scala.util.Random.nextDouble
    val it = dist.iterator
    var accum = 0.0
    while (it.hasNext) {
      val (item, itemProb) = it.next
      accum += itemProb
      if (accum >= p)
        return item
    }
    sys.error(f"this should never happen")
  }
  def get_zero_mat(n: Int, k: Int): Vector [Vector[Double]]={
    val zero_mat= Vector.fill(n, k)(0.0)
    return zero_mat
  }
  def projection_rdd(sc: SparkContext, n: Long, k: Long, q:Double):Unit = {
    // create nXk dimensional matrix
    val zero_matrix = get_zero_mat(n.toInt, k.toInt)
    val zero_rdd = sc.parallelize(zero_matrix)
    val constant = math.sqrt(1.0/k*q)
    val probs = List(q/2,q/2, 1 -q)
    val options = List(constant, -constant, 0)
    val dist = (options zip probs) toMap

    //val dist = Map(options(0)-> probs(0), options(1) -> probs(1), options(2)-> probs(2))

    //np.random.choice([konstant, -konstant, 0], size=(k,n), p=[q/2, q/2,1-q])
    val vecs = Vector.fill(n.toInt)(sample(dist))



  }
  def main(args: Array[String]) {
    if (args.length != 2) {
      System.err.println("Usage: SparkApp <input> <lower_dimension>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Randomized LA")
    val sc = new SparkContext(conf)
    val lowerDim = args(1)

    // Load and parse the data file.
    val rows = sc.textFile(args(0)).map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.dense(values)
    }
    val mat = new RowMatrix(rows)
    val nrows = mat.numRows()
    val ncols = mat.numCols()
    val proj = projection_rdd(nrows, ncols, q=0.1)

    sc.stop()
  }
}

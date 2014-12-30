package randla
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Matrices, Vectors, Vector}
import org.apache.spark.rdd.RDD
import breeze.linalg.qr.QR

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
    sys.error("this should never happen")
  }
  def get_zero_mat(n: Int, k: Int): Array [Array[Double]]={
    val zero_mat= Array.fill(n, k)(0.0)
    return zero_mat
  }


  def projection_rdd(sc: SparkContext, n: Long, k: Long, q:Double):RowMatrix = {
    // create kXn dimensional matrix
    val zero_matrix = get_zero_mat(k.toInt, n.toInt)
    val zero_rdd = sc.parallelize(zero_matrix)
    val constant = math.sqrt(1.0/k*q)
    val probs = List(q/2,q/2, 1 -q)
    val options = List(constant, -constant, 0)
    val dist = (options zip probs) toMap

    //val dist = Map(options(0)-> probs(0), options(1) -> probs(1), options(2)-> probs(2))

    //np.random.choice([konstant, -konstant, 0], size=(k,n), p=[q/2, q/2,1-q])
    //val vecs = Vector.fill(n.toInt)(sample(dist))
    val projection_matrix = zero_rdd.map(_ => Array.fill(n.toInt)(sample(dist))).map(x => Vectors.dense(x))
    val projMat = new RowMatrix(projection_matrix)
    return projMat

  }
  def transpose(m: Array[Array[Double]]): Array[Array[Double]] = {
    (for {
      c <- m(0).indices
    } yield m.map(_(c)) ).toArray
  }

  def multiply(P: RowMatrix, rows:RDD[Vector]): RowMatrix = {
    val ma = rows.map(_.toArray).take(rows.count.toInt)
    val localMat = Matrices.dense( rows.count.toInt,
      rows.take(1)(0).size,
      transpose(ma).flatten)


    return P.multiply(localMat)
  }

  def main(args: Array[String]) {
    if (args.length != 2) {
      System.err.println("Usage: SparkApp <input> <lower_dimension>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Randomized LA")
    val sc = new SparkContext(conf)
    val lowerDim = args(1).toInt

    // Load and parse the data file.
    val rows = sc.textFile(args(0)).map { line =>
      val values = line.split(' ').map(_.toDouble)
      Vectors.dense(values)
    }
    val mat = new RowMatrix(rows)
    val nrows = mat.numRows()
    val ncols = mat.numCols()
    val projection = projection_rdd(sc, nrows, lowerDim, q=0.1)
    val proj_mat = multiply(projection, rows)
    //val proj_rows = proj_mat.
    println(proj_mat.numRows() + " " +proj_mat.numCols())
    val (_q,_r) = qr_factorization.qr_factor(proj_mat, proj_mat.numRows().toInt, proj_mat.numCols().toInt)
    println(_r)

    sc.stop()
  }
}

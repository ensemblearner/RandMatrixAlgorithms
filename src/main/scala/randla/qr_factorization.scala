package randla

import breeze.linalg.{qr, DenseMatrix}
import breeze.linalg.qr.QR
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
 * Created by mohit on 12/29/14.
 */
object qr_factorization {
  def qr_factor(A: RowMatrix,nrows: Int, ncols: Int):(Any, Any) = {
    val rows = A.rows.map(_.toArray).collect().flatten
    val dense_matrix = DenseMatrix.fill(nrows, ncols)(rows)
    println(dense_matrix)
    val QR(_Q, _R) = qr(dense_matrix)
    return (_Q,_R)

  }

}

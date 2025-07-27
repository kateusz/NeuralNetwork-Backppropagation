namespace Engine.LossFunctions;

/// <summary>
/// Mean Squared Error - najpopularniejsza funkcja straty dla problemów regresji.
/// Wzór: MSE = (1/2) × Σ(target - prediction)²
/// Gradient: ∂MSE/∂prediction = -(target - prediction)
/// 
/// Dzielimy przez 2 żeby pochodna była ładniejsza (znika dwójka z różniczkowania x²).
/// </summary>
public class MeanSquaredError : ILossFunction
{
    public string Name => "Mean Squared Error";

    /// <summary>
    /// Oblicza MSE dla całego wektora przewidywań.
    /// </summary>
    public double CalculateLoss(double[] predicted, double[] target)
    {
        if (predicted.Length != target.Length)
            throw new ArgumentException("Predicted and target arrays must have the same length");

        double sumSquaredErrors = 0.0;
            
        for (int i = 0; i < predicted.Length; i++)
        {
            double error = target[i] - predicted[i];
            sumSquaredErrors += error * error;
        }
            
        // Dzielimy przez 2 dla ładnej pochodnej, przez length dla średniej
        return sumSquaredErrors / (2.0 * predicted.Length);
    }

    /// <summary>
    /// Oblicza gradient MSE względem każdego przewidywania.
    /// Gradient MSE = -(target - prediction) dla każdego elementu.
    /// </summary>
    public double[] CalculateDerivative(double[] predicted, double[] target)
    {
        if (predicted.Length != target.Length)
            throw new ArgumentException("Predicted and target arrays must have the same length");

        double[] gradient = new double[predicted.Length];
            
        for (int i = 0; i < predicted.Length; i++)
        {
            // Pochodna (1/2)(target - prediction)² = -(target - prediction)
            gradient[i] = -(target[i] - predicted[i]) / predicted.Length;
        }
            
        return gradient;
    }
}
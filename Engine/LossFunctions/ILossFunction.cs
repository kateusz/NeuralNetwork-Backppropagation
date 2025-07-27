namespace Engine.LossFunctions;

// <summary>
    /// Interfejs dla funkcji straty (loss functions).
    /// Funkcja straty mierzy, jak bardzo przewidywania sieci różnią się od prawdziwych wartości.
    /// </summary>
    public interface ILossFunction
    {
        /// <summary>
        /// Oblicza wartość funkcji straty.
        /// </summary>
        /// <param name="predicted">Przewidywania sieci</param>
        /// <param name="target">Prawdziwe wartości docelowe</param>
        /// <returns>Wartość straty</returns>
        double CalculateLoss(double[] predicted, double[] target);
        
        /// <summary>
        /// Oblicza gradient funkcji straty względem przewidywań.
        /// Ten gradient jest używany jako punkt startowy dla backpropagation.
        /// </summary>
        /// <param name="predicted">Przewidywania sieci</param>
        /// <param name="target">Prawdziwe wartości docelowe</param>
        /// <returns>Gradient straty względem każdego przewidywania</returns>
        double[] CalculateDerivative(double[] predicted, double[] target);
        
        /// <summary>
        /// Nazwa funkcji straty.
        /// </summary>
        string Name { get; }
    }
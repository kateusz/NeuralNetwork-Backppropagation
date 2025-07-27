namespace Engine.Activations;

// <summary>
    /// Funkcja aktywacji Sigmoid.
    /// Mapuje wartości rzeczywiste na przedział (0, 1).
    /// Wzór: σ(x) = 1 / (1 + exp(-x))
    /// Pochodna: σ'(x) = σ(x) × (1 - σ(x))
    /// </summary>
    public class Sigmoid : IActivationFunction
    {
        public string Name => "Sigmoid";

        /// <summary>
        /// Oblicza wartość funkcji sigmoid.
        /// Uwaga: używamy Math.Exp(-x) zamiast 1/Math.Exp(x) dla stabilności numerycznej.
        /// </summary>
        public double Activate(double x)
        {
            // Zabezpieczenie przed overflow - jeśli x jest bardzo duże ujemne,
            // exp(-x) będzie bardzo duże, więc ograniczamy x
            if (x < -500) return 0.0;
            if (x > 500) return 1.0;
            
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Oblicza pochodną sigmoid używając właściwości: σ'(x) = σ(x) × (1 - σ(x)).
        /// To jest bardzo efektywne, bo pochodną można obliczyć z już znanej wartości sigmoid.
        /// </summary>
        public double Derivative(double x)
        {
            double sigmoid = Activate(x);
            return sigmoid * (1.0 - sigmoid);
        }
    }
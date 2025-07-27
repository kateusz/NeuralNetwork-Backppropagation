namespace Engine.Activations;

/// <summary>
/// Funkcja aktywacji Tanh (tangens hiperboliczny).
/// Mapuje wartości na przedział (-1, 1), co często daje lepsze wyniki niż sigmoid.
/// Wzór: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// Pochodna: tanh'(x) = 1 - tanh²(x)
/// </summary>
public class Tanh : IActivationFunction
{
    public string Name => "Tanh";

    /// <summary>
    /// Oblicza tangens hiperboliczny.
    /// Używamy wbudowanej funkcji Math.Tanh dla stabilności numerycznej.
    /// </summary>
    public double Activate(double x)
    {
        // Math.Tanh jest już zoptymalizowane pod kątem stabilności numerycznej
        return Math.Tanh(x);
    }

    /// <summary>
    /// Pochodna tanh używa pięknej właściwości: tanh'(x) = 1 - tanh²(x).
    /// Podobnie jak w sigmoid, możemy użyć już obliczonej wartości funkcji.
    /// </summary>
    public double Derivative(double x)
    {
        double tanh = Activate(x);
        return 1.0 - tanh * tanh;
    }
}
namespace Engine.Activations;

/// <summary>
/// Funkcja aktywacji ReLU (Rectified Linear Unit).
/// Bardzo popularna w głębokich sieciach ze względu na prostotę i brak problemu vanishing gradient.
/// Wzór: f(x) = max(0, x)
/// Pochodna: f'(x) = 1 jeśli x > 0, inaczej 0
/// </summary>
public class ReLU : IActivationFunction
{
    public string Name => "ReLU";

    /// <summary>
    /// Implementacja ReLU - zwraca maksimum z 0 i x.
    /// </summary>
    public double Activate(double x)
    {
        return Math.Max(0.0, x);
    }

    /// <summary>
    /// Pochodna ReLU - bardzo prosta: 1 dla x > 0, 0 w przeciwnym przypadku.
    /// Uwaga: technicznie w x = 0 pochodna nie istnieje, ale przyjmujemy 0.
    /// </summary>
    public double Derivative(double x)
    {
        return x > 0 ? 1.0 : 0.0;
    }
}
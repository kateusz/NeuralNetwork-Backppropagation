namespace Engine.Activations;

/// <summary>
/// Interfejs dla funkcji aktywacji w sieci neuronowej.
/// Każda funkcja aktywacji musi implementować zarówno funkcję przodu jak i jej pochodną.
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// Oblicza wartość funkcji aktywacji dla podanej wartości.
    /// </summary>
    /// <param name="x">Wartość wejściowa (net input)</param>
    /// <returns>Aktywacja neuronu</returns>
    double Activate(double x);
        
    /// <summary>
    /// Oblicza pochodną funkcji aktywacji dla podanej wartości.
    /// </summary>
    /// <param name="x">Wartość wejściowa (net input)</param>
    /// <returns>Pochodna funkcji aktywacji</returns>
    double Derivative(double x);
        
    /// <summary>
    /// Nazwa funkcji aktywacji dla celów logowania/debugowania.
    /// </summary>
    string Name { get; }
}
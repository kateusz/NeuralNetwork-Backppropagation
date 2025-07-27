using Engine.Activations;

namespace Engine.Core;

/// <summary>
/// Pojedyncza warstwa sieci neuronowej implementująca forward propagation.
/// 
/// Matematycznie reprezentuje transformację: a^(l) = f(W^(l) × a^(l-1) + b^(l))
/// gdzie:
/// - W^(l) to macierz wag [output_neurons × input_neurons] (Matrix class)
/// - a^(l-1) to wektor wejściowy [input_neurons]  
/// - b^(l) to wektor biasów [output_neurons]
/// - f() to funkcja aktywacji
/// - a^(l) to wektor wyjściowy [output_neurons]
/// </summary>
public class Layer
{
    /// <summary>
    /// Macierz wag połączeń [output_neurons × input_neurons].
    /// </summary>
    private readonly Matrix _weights;
    
    /// <summary>
    /// Wektor biasów [output_neurons].
    /// biases[i] to bias dodawany do i-tego neuronu wyjściowego.
    /// </summary>
    private readonly double[] _biases;
    
    /// <summary>
    /// Funkcja aktywacji stosowana do każdego neuronu w warstwie.
    /// </summary>
    private readonly IActivationFunction _activation;
    
    #region Public Properties
    
    // Cached values dla backward propagation
    /// <summary>
    /// Cache ostatnich wejść - potrzebne dla obliczania gradientów wag.
    /// </summary>
    public double[]? LastInputs { get; private set; }
    
    /// <summary>
    /// Cache ostatnich net inputs (przed aktywacją) - potrzebne dla pochodnych aktywacji.
    /// </summary>
    public double[]? LastNetInputs { get; private set; }
    
    /// <summary>
    /// Cache ostatnich wyjść (po aktywacji) - potrzebne dla następnej warstwy.
    /// </summary>
    public double[]? LastOutputs { get; private set; }
    
    /// <summary>
    /// Liczba neuronów wejściowych (rozmiar wektora wejściowego).
    /// </summary>
    public int InputSize { get; }
    
    /// <summary>
    /// Liczba neuronów wyjściowych (rozmiar wektora wyjściowego).
    /// </summary>
    public int OutputSize { get; }
    
    /// <summary>
    /// Nazwa funkcji aktywacji używanej w tej warstwie.
    /// </summary>
    public string ActivationName => _activation.Name;
    
    public Matrix WeightsMatrix => _weights;
    
    #endregion
    
    /// <summary>
    /// Tworzy nową warstwę sieci neuronowej.
    /// </summary>
    /// <param name="inputSize">Liczba neuronów wejściowych</param>
    /// <param name="outputSize">Liczba neuronów wyjściowych</param>
    /// <param name="activation">Funkcja aktywacji</param>
    /// <exception cref="ArgumentException">Gdy rozmiary są nieprawidłowe</exception>
    /// <exception cref="ArgumentNullException">Gdy funkcja aktywacji jest null</exception>
    public Layer(int inputSize, int outputSize, IActivationFunction activation)
    {
        if (inputSize <= 0)
            throw new ArgumentException("Input size must be positive", nameof(inputSize));
        if (outputSize <= 0)
            throw new ArgumentException("Output size must be positive", nameof(outputSize));

        InputSize = inputSize;
        OutputSize = outputSize;
        _activation = activation ?? throw new ArgumentNullException(nameof(activation));
        
        // Używamy Matrix.Random dla automatic Xavier initialization
        _weights = Matrix.Random(outputSize, inputSize, seed: 42);
        _biases = new double[outputSize]; // Initialized to zeros
    }

    #region Core Methods
    
    /// <summary>
    /// Forward propagation przez warstwę.
    /// 
    /// Implementuje matematyczną transformację:
    /// 1. z^(l) = W^(l) × a^(l-1) + b^(l)  (linear transformation)
    /// 2. a^(l) = f(z^(l))                  (non-linear activation)
    /// 
    /// Wykorzystuje Matrix.MultiplyVector() z SIMD optimization.
    /// Wyniki są cache'owane dla późniejszego backward pass.
    /// </summary>
    /// <param name="inputs">Wektor wejściowy [InputSize]</param>
    /// <returns>Wektor wyjściowy [OutputSize]</returns>
    /// <exception cref="ArgumentNullException">Gdy inputs jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiar inputs nie pasuje do InputSize</exception>
    public double[] Forward(double[] inputs)
    {
        // Walidacja wejścia
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != InputSize)
            throw new ArgumentException(
                $"Input size {inputs.Length} doesn't match layer input size {InputSize}", 
                nameof(inputs));
        
        // Cache inputs dla backward pass (defensive copy)
        LastInputs = new double[InputSize];
        Array.Copy(inputs, LastInputs, InputSize);
        
        // KROK 1: Linear transformation z = W × a + b
        // Używamy Matrix.MultiplyVector() z SIMD optimization
        var netInputs = _weights.MultiplyVector(inputs);
        
        // Dodaj biases element-wise
        for (int i = 0; i < OutputSize; i++)
        {
            netInputs[i] += _biases[i];
        }
        
        // Cache net inputs dla backward pass
        LastNetInputs = new double[OutputSize];
        Array.Copy(netInputs, LastNetInputs, OutputSize);
        
        // KROK 2: Non-linear activation a = f(z)
        var outputs = ApplyActivation(netInputs);
        
        // Cache outputs dla backward pass
        LastOutputs = new double[OutputSize];
        Array.Copy(outputs, LastOutputs, OutputSize);
        
        return outputs;
    }
    
    #endregion
    
    #region Private Core Methods
    
    /// <summary>
    /// Stosuje funkcję aktywacji do każdego elementu net inputs.
    /// </summary>
    /// <param name="netInputs">Net inputs (przed aktywacją)</param>
    /// <returns>Activated outputs</returns>
    private double[] ApplyActivation(double[] netInputs)
    {
        var outputs = new double[OutputSize];
        
        // Zastosuj funkcję aktywacji element-wise
        for (int i = 0; i < OutputSize; i++)
        {
            outputs[i] = _activation.Activate(netInputs[i]);
        }
        
        return outputs;
    }
    
    #endregion
    
    #region Weight and Bias Management
    
    /// <summary>
    /// Ustawia wagi warstwy z Matrix (głównie do testów).
    /// </summary>
    /// <param name="weights">Nowa macierz wag [OutputSize × InputSize]</param>
    /// <exception cref="ArgumentNullException">Gdy weights jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiary macierzy nie pasują</exception>
    public void SetWeights(Matrix weights)
    {
        if (weights == null)
            throw new ArgumentNullException(nameof(weights));
        if (weights.Rows != OutputSize || weights.Cols != InputSize)
            throw new ArgumentException(
                $"Weight matrix size {weights.Rows}×{weights.Cols} " +
                $"doesn't match layer size {OutputSize}×{InputSize}");
        
        // Copy data from provided matrix to our internal matrix
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                _weights[i, j] = weights[i, j];
            }
        }
    }
    
    /// <summary>
    /// Ustawia biasy warstwy (głównie do testów).
    /// </summary>
    /// <param name="biases">Nowy wektor biasów [OutputSize]</param>
    /// <exception cref="ArgumentNullException">Gdy biases jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiar wektora nie pasuje</exception>
    public void SetBiases(double[] biases)
    {
        if (biases == null)
            throw new ArgumentNullException(nameof(biases));
        if (biases.Length != OutputSize)
            throw new ArgumentException(
                $"Bias vector size {biases.Length} doesn't match output size {OutputSize}");
        
        Array.Copy(biases, _biases, OutputSize);
    }
    
    /// <summary>
    /// Pobiera kopię macierzy wag jako Matrix (głównie do testów i debugowania).
    /// </summary>
    /// <returns>Kopia macierzy wag jako Matrix</returns>
    public Matrix GetWeightsMatrix()
    {
        // Matrix class already provides defensive copying in its operations
        var copy = new Matrix(OutputSize, InputSize);
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                copy[i, j] = _weights[i, j];
            }
        }
        return copy;
    }
    
    /// <summary>
    /// Pobiera kopię wektora biasów (głównie do testów i debugowania).
    /// </summary>
    /// <returns>Kopia wektora biasów</returns>
    public double[] GetBiases()
    {
        var copy = new double[OutputSize];
        Array.Copy(_biases, copy, OutputSize);
        return copy;
    }
    
    #endregion
    
    #region Advanced Matrix Operations
    
    /// <summary>
    /// Oblicza transpozycję macierzy wag (potrzebne dla backward propagation).
    /// </summary>
    /// <returns>Transponowana macierz wag [InputSize × OutputSize]</returns>
    public Matrix GetWeightsTranspose()
    {
        return _weights.Transpose();
    }
    
    /// <summary>
    /// Mnoży transpozycję wag przez wektor (common operation w backpropagation).
    /// Implementuje: W^T × vector, gdzie W^T ma wymiary [InputSize × OutputSize].
    /// </summary>
    /// <param name="vector">Wektor do pomnożenia [OutputSize]</param>
    /// <returns>Wynik mnożenia [InputSize]</returns>
    /// <exception cref="ArgumentNullException">Gdy vector jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiar wektora nie pasuje</exception>
    public double[] MultiplyWeightsTranspose(double[] vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));
        if (vector.Length != OutputSize)
            throw new ArgumentException(
                $"Vector length {vector.Length} doesn't match output size {OutputSize}",
                nameof(vector));
        
        var weightsTranspose = _weights.Transpose();
        return weightsTranspose.MultiplyVector(vector);
    }
    
    #endregion
    
    #region Diagnostics and Debugging
    
    /// <summary>
    /// Sprawdza czy warstwa ma cache'owane wartości z ostatniego Forward pass.
    /// </summary>
    public bool HasCachedValues => LastInputs != null && LastNetInputs != null && LastOutputs != null;
    
    /// <summary>
    /// Zwraca podstawowe statystyki wag dla diagnostyki.
    /// </summary>
    /// <returns>String z statystykami wag</returns>
    public string GetWeightStatistics()
    {
        var weights1D = _weights.ToArray1D();
        
        var mean = weights1D.Average();
        var variance = weights1D.Select(w => (w - mean) * (w - mean)).Average();
        var stdDev = Math.Sqrt(variance);
        var min = weights1D.Min();
        var max = weights1D.Max();
        
        return $"Weights: μ={mean:F4}, σ={stdDev:F4}, range=[{min:F4}, {max:F4}]";
    }
    
    /// <summary>
    /// Zwraca string reprezentację warstwy dla debugowania.
    /// </summary>
    public override string ToString()
    {
        return $"Layer({InputSize} → {OutputSize}, {ActivationName})";
    }
    
    #endregion
}
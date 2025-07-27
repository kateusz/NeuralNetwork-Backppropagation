using System.Numerics;
using Engine.Activations;

namespace Engine.Core;

/// <summary>
/// Pojedyncza warstwa sieci neuronowej implementująca forward propagation.
/// 
/// Matematycznie reprezentuje transformację: a^(l) = f(W^(l) × a^(l-1) + b^(l))
/// gdzie:
/// - W^(l) to macierz wag [output_neurons × input_neurons]
/// - a^(l-1) to wektor wejściowy [input_neurons]  
/// - b^(l) to wektor biasów [output_neurons]
/// - f() to funkcja aktywacji
/// - a^(l) to wektor wyjściowy [output_neurons]
/// </summary>
public class Layer
{
    /// <summary>
    /// Macierz wag połączeń [output_neurons × input_neurons].
    /// weights[i,j] reprezentuje siłę połączenia od input j do output neuron i.
    /// </summary>
    private readonly double[,] _weights;
    
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
        
        // Inicjalizuj macierze
        _weights = new double[outputSize, inputSize];
        _biases = new double[outputSize];
        
        // Losowa inicjalizacja wag
        InitializeWeights();
    }

    #region Core Methods
    
    /// <summary>
    /// Forward propagation przez warstwę.
    /// 
    /// Implementuje matematyczną transformację:
    /// 1. z^(l) = W^(l) × a^(l-1) + b^(l)  (linear transformation)
    /// 2. a^(l) = f(z^(l))                  (non-linear activation)
    /// 
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
        
        // Cache inputs dla backward pass
        LastInputs = new double[InputSize];
        Array.Copy(inputs, LastInputs, InputSize);
        
        // KROK 1: Linear transformation z = W × a + b
        var netInputs = ComputeLinearTransformation(inputs);
        
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
    /// Oblicza linear transformation: z = W × a + b
    /// Używa System.Numerics dla optymalizacji SIMD gdzie to możliwe.
    /// </summary>
    /// <param name="inputs">Wektor wejściowy</param>
    /// <returns>Net inputs (przed aktywacją)</returns>
    private double[] ComputeLinearTransformation(double[] inputs)
    {
        var netInputs = new double[OutputSize];
        
        // Matrix-vector multiplication z SIMD optimization
        for (int i = 0; i < OutputSize; i++)
        {
            double sum = 0.0;
            
            // Użyj Vector<double> dla SIMD optimization gdy to możliwe
            int vectorLength = Vector<double>.Count;
            int vectorizedLength = (InputSize / vectorLength) * vectorLength;
            
            // SIMD-optimized portion
            for (int j = 0; j < vectorizedLength; j += vectorLength)
            {
                // Pobierz wagi dla tego neuronu
                var weights = new double[vectorLength];
                for (int k = 0; k < vectorLength; k++)
                {
                    weights[k] = _weights[i, j + k];
                }
                
                // Pobierz odpowiadające inputs
                var inputSegment = new double[vectorLength];
                Array.Copy(inputs, j, inputSegment, 0, vectorLength);
                
                // SIMD dot product
                var weightVector = new Vector<double>(weights);
                var inputVector = new Vector<double>(inputSegment);
                sum += Vector.Dot(weightVector, inputVector);
            }
            
            // Handle remaining elements (non-vectorized)
            for (int j = vectorizedLength; j < InputSize; j++)
            {
                sum += _weights[i, j] * inputs[j];
            }
            
            // Dodaj bias
            netInputs[i] = sum + _biases[i];
        }
        
        return netInputs;
    }
    
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
    
    #region Weight Management
    
    /// <summary>
    /// Inicjalizuje wagi losowo używając uproszczonej inicjalizacji Xavier.
    /// Wagi są losowane z rozkładu normalnego przeskalowanego według liczby połączeń.
    /// </summary>
    private void InitializeWeights()
    {
        var random = new Random(42); // Fixed seed dla reproducibility w testach
        
        // Xavier initialization: wagi ~ N(0, 1/n) gdzie n = InputSize
        double scale = 1.0 / Math.Sqrt(InputSize);
        
        for (int i = 0; i < OutputSize; i++)
        {
            for (int j = 0; j < InputSize; j++)
            {
                // Box-Muller transformation dla rozkładu normalnego
                _weights[i, j] = GenerateGaussianRandom(random) * scale;
            }
            
            // Biasy inicjalizujemy zerem
            _biases[i] = 0.0;
        }
    }
    
    /// <summary>
    /// Generuje liczbę losową z rozkładu normalnego N(0,1).
    /// </summary>
    private static double GenerateGaussianRandom(Random random)
    {
        // Box-Muller transformation
        static double NextGaussian(Random rnd)
        {
            double u1 = 1.0 - rnd.NextDouble();
            double u2 = 1.0 - rnd.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
        
        return NextGaussian(random);
    }
    
    #endregion
    
    #region Testing Support Methods
    
    /// <summary>
    /// Ustawia wagi warstwy (głównie do testów).
    /// </summary>
    /// <param name="weights">Nowa macierz wag [OutputSize × InputSize]</param>
    /// <exception cref="ArgumentNullException">Gdy weights jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiary macierzy nie pasują</exception>
    public void SetWeights(double[,] weights)
    {
        if (weights == null)
            throw new ArgumentNullException(nameof(weights));
        if (weights.GetLength(0) != OutputSize || weights.GetLength(1) != InputSize)
            throw new ArgumentException(
                $"Weight matrix size [{weights.GetLength(0)},{weights.GetLength(1)}] " +
                $"doesn't match layer size [{OutputSize},{InputSize}]");
        
        Array.Copy(weights, _weights, weights.Length);
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
    /// Pobiera kopię macierzy wag (głównie do testów i debugowania).
    /// </summary>
    /// <returns>Kopia macierzy wag</returns>
    public double[,] GetWeights()
    {
        var copy = new double[OutputSize, InputSize];
        Array.Copy(_weights, copy, _weights.Length);
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
    
    #region Diagnostics and Debugging
    
    /// <summary>
    /// Sprawdza czy warstwa ma cache'owane wartości z ostatniego Forward pass.
    /// </summary>
    public bool HasCachedValues => LastInputs != null && LastNetInputs != null && LastOutputs != null;
    
    /// <summary>
    /// Zwraca string reprezentację warstwy dla debugowania.
    /// </summary>
    public override string ToString()
    {
        return $"Layer({InputSize} → {OutputSize}, {ActivationName})";
    }
    
    #endregion
}
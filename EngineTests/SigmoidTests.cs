using Engine.Activations;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Testy dla funkcji aktywacji Sigmoid.
/// Sigmoid jest jedną z najważniejszych funkcji aktywacji w historii sieci neuronowych.
/// Te testy sprawdzają zarówno poprawność matematyczną jak i przypadki brzegowe.
/// </summary>
public class SigmoidTests
{
    private readonly Sigmoid sigmoid = new Sigmoid();
    private const double Tolerance = 1e-10; // Tolerancja dla porównań zmiennoprzecinkowych

    /// <summary>
    /// Sprawdza, czy sigmoid(0) = 0.5.
    /// To jest fundamentalna właściwość funkcji sigmoid - w punkcie zerowym powinna zwracać dokładnie 0.5.
    /// Ta właściwość jest ważna, bo oznacza że sigmoid jest symetryczny względem punktu (0, 0.5).
    /// </summary>
    [Fact]
    public void Sigmoid_AtZero_ShouldReturn_0_5()
    {
        var result = sigmoid.Activate(0.0);
        result.ShouldBe(0.5, Tolerance);
    }

    /// <summary>
    /// Sprawdza asymptotyczne zachowanie sigmoid dla dużych wartości dodatnich.
    /// Sigmoid powinien dążyć do 1, ale nigdy jej nie osiągnąć.
    /// To jest ważne dla stabilności numerycznej - funkcja nie powinna zwracać dokładnie 1.0.
    /// </summary>
    [Fact]
    public void Sigmoid_AtLargePositiveValue_ShouldApproach_1()
    {
        var result = sigmoid.Activate(10.0);
        result.ShouldBeGreaterThan(0.99);
        result.ShouldBeLessThan(1.0);
    }

    /// <summary>
    /// Sprawdza asymptotyczne zachowanie sigmoid dla dużych wartości ujemnych.
    /// Sigmoid powinien dążyć do 0, ale nigdy jej nie osiągnąć.
    /// To zapewnia, że funkcja pozostaje różniczkowalna w całej dziedzinie.
    /// </summary>
    [Fact]
    public void Sigmoid_AtLargeNegativeValue_ShouldApproach_0()
    {
        var result = sigmoid.Activate(-10.0);
        result.ShouldBeLessThan(0.01);
        result.ShouldBeGreaterThan(0.0);
    }

    /// <summary>
    /// Weryfikuje, że sigmoid jest funkcją ściśle rosnącą.
    /// To jest kluczowa właściwość - sigmoid musi być monotoniczny, żeby zachować porządek w danych.
    /// Jeśli x1 < x2, to sigmoid(x1) < sigmoid(x2). Bez tej właściwości funkcja byłaby bezużyteczna.
    /// </summary>
    [Fact]
    public void Sigmoid_IsMonotonicIncreasing()
    {
        var testPoints = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };
        var results = new double[testPoints.Length];
            
        for (int i = 0; i < testPoints.Length; i++)
        {
            results[i] = sigmoid.Activate(testPoints[i]);
        }
            
        // Sprawdzamy, że każdy kolejny wynik jest większy od poprzedniego
        for (int i = 0; i < results.Length - 1; i++)
        {
            results[i + 1].ShouldBeGreaterThan(results[i], 
                $"Sigmoid should be increasing: f({testPoints[i + 1]}) should be > f({testPoints[i]})");
        }
    }

    /// <summary>
    /// Sprawdza, czy pochodna sigmoid w punkcie 0 wynosi 0.25.
    /// σ'(0) = σ(0) × (1 - σ(0)) = 0.5 × 0.5 = 0.25
    /// To jest maksymalna wartość pochodnej sigmoid - ważne dla zrozumienia szybkości uczenia.
    /// </summary>
    [Fact]
    public void Sigmoid_Derivative_AtZero_ShouldReturn_0_25()
    {
        var result = sigmoid.Derivative(0.0);
        result.ShouldBe(0.25, Tolerance);
    }

    /// <summary>
    /// Weryfikuje poprawność implementacji pochodnej przez porównanie z gradientem numerycznym.
    /// To jest kluczowy test - jeśli analityczna pochodna nie zgadza się z numeryczną,
    /// oznacza to błąd w implementacji, który uniemożliwi działanie backpropagation.
    /// Używamy metody różnic skończonych: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    /// </summary>
    [Fact]
    public void Sigmoid_Derivative_MatchesNumericalDerivative()
    {
        var testPoints = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };
        const double h = 1e-8; // Bardzo mały krok dla różnicy skończonej
            
        foreach (var x in testPoints)
        {
            // Pochodna numeryczna: (f(x+h) - f(x-h)) / (2h)
            var numericalDerivative = (sigmoid.Activate(x + h) - sigmoid.Activate(x - h)) / (2 * h);
            var analyticalDerivative = sigmoid.Derivative(x);
                
            analyticalDerivative.ShouldBe(numericalDerivative, 1e-6, 
                $"Analytical derivative should match numerical derivative at x = {x}");
        }
    }

    /// <summary>
    /// Testuje stabilność numeryczną dla ekstremalnych wartości wejściowych.
    /// Sigmoid musi gracefully handle bardzo duże i bardzo małe wartości bez powodowania overflow/underflow.
    /// To jest praktyczny test - w rzeczywistych sieciach neuronowych często pojawiają się ekstremalne wartości.
    /// </summary>
    [Fact]
    public void Sigmoid_HandlesExtremeValues()
    {
        sigmoid.Activate(1000).ShouldBe(1.0, Tolerance);
        sigmoid.Activate(-1000).ShouldBe(0.0, Tolerance);
    }

    /// <summary>
    /// Sprawdza, czy implementacja używa właściwej właściwości matematycznej pochodnej sigmoid.
    /// Pochodna sigmoid ma piękną właściwość: σ'(x) = σ(x) × (1 - σ(x))
    /// To oznacza, że możemy obliczyć pochodną z już znanej wartości funkcji, co jest bardzo efektywne.
    /// </summary>
    [Fact]
    public void Sigmoid_Derivative_UsesOptimizedFormula()
    {
        const double x = 1.5;
        var sigmoidValue = sigmoid.Activate(x);
        var expectedDerivative = sigmoidValue * (1.0 - sigmoidValue);
        var actualDerivative = sigmoid.Derivative(x);
            
        actualDerivative.ShouldBe(expectedDerivative, Tolerance,
            "Derivative should use the optimized formula σ'(x) = σ(x) × (1 - σ(x))");
    }
}
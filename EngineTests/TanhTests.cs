using Engine.Activations;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Testy dla funkcji aktywacji Tanh (tangens hiperboliczny).
/// Tanh ma podobne właściwości do sigmoid, ale z zakresem (-1, 1) zamiast (0, 1),
/// co często daje lepsze wyniki w uczeniu.
/// </summary>
public class TanhTests
{
    private readonly Tanh tanh = new Tanh();
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Sprawdza, że tanh(0) = 0.
    /// To jest podstawowa właściwość funkcji nieparzystej - przechodzi przez początek układu współrzędnych.
    /// Ważne dla symetrii funkcji i zerowego bias w punkcie neutralnym.
    /// </summary>
    [Fact]
    public void Tanh_AtZero_ShouldReturnZero()
    {
        tanh.Activate(0.0).ShouldBe(0.0, Tolerance);
    }

    /// <summary>
    /// Weryfikuje, że tanh jest funkcją nieparzystą: tanh(-x) = -tanh(x).
    /// Ta właściwość oznacza symetrię względem początku układu współrzędnych.
    /// Jest matematycznie fundamentalna i ważna dla stabilności uczenia.
    /// </summary>
    [Fact]
    public void Tanh_IsOddFunction()
    {
        var testValues = new[] { 0.5, 1.0, 2.0, 3.0 };
            
        foreach (var x in testValues)
        {
            var posResult = tanh.Activate(x);
            var negResult = tanh.Activate(-x);
                
            negResult.ShouldBe(-posResult, Tolerance,
                $"Tanh should be an odd function: tanh(-{x}) should equal -tanh({x})");
        }
    }

    /// <summary>
    /// Sprawdza, że tanh zawsze zwraca wartości z przedziału [-1, 1].
    /// To jest definicyjny zakres wartości tanh, ważny dla boundedness of activations.
    /// Zapewnia, że aktywacje nie explodują do nieskończoności.
    /// </summary>
    [Fact]
    public void Tanh_RangeIsBetween_MinusOne_And_One()
    {
        var testValues = new[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
            
        foreach (var x in testValues)
        {
            var result = tanh.Activate(x);
            result.ShouldBeGreaterThanOrEqualTo(-1.0, 
                $"Tanh result should be ≥ -1 for input {x}");
            result.ShouldBeLessThanOrEqualTo(1.0, 
                $"Tanh result should be ≤ 1 for input {x}");
        }
    }

    /// <summary>
    /// Weryfikuje, że pochodna tanh w punkcie zero wynosi 1.
    /// tanh'(0) = 1 - tanh²(0) = 1 - 0² = 1.
    /// To jest maksymalna wartość pochodnej tanh, ważna dla maksymalnej szybkości uczenia.
    /// </summary>
    [Fact]
    public void Tanh_Derivative_AtZero_ShouldReturnOne()
    {
        tanh.Derivative(0.0).ShouldBe(1.0, Tolerance);
    }

    /// <summary>
    /// Porównuje analityczną pochodną tanh z gradientem numerycznym.
    /// Sprawdza poprawność implementacji wzoru: tanh'(x) = 1 - tanh²(x).
    /// Test kluczowy dla poprawności backpropagation - błędna pochodna uniemożliwi uczenie.
    /// </summary>
    [Fact]
    public void Tanh_Derivative_MatchesNumericalDerivative()
    {
        var testPoints = new[] { -2.0, -1.0, 0.0, 1.0, 2.0 };
        const double h = 1e-8;
            
        foreach (var x in testPoints)
        {
            var numericalDerivative = (tanh.Activate(x + h) - tanh.Activate(x - h)) / (2 * h);
            var analyticalDerivative = tanh.Derivative(x);
                
            analyticalDerivative.ShouldBe(numericalDerivative, 1e-6,
                $"Analytical derivative should match numerical derivative at x = {x}");
        }
    }

    /// <summary>
    /// Sprawdza, czy implementacja używa właściwej właściwości matematycznej pochodnej tanh.
    /// Pochodna tanh: tanh'(x) = 1 - tanh²(x), co pozwala obliczyć pochodną z już znanej wartości funkcji.
    /// Ta optymalizacja jest ważna dla wydajności w dużych sieciach.
    /// </summary>
    [Fact]
    public void Tanh_Derivative_UsesOptimizedFormula()
    {
        const double x = 1.5;
        var tanhValue = tanh.Activate(x);
        var expectedDerivative = 1.0 - tanhValue * tanhValue;
        var actualDerivative = tanh.Derivative(x);
            
        actualDerivative.ShouldBe(expectedDerivative, Tolerance,
            "Derivative should use the optimized formula tanh'(x) = 1 - tanh²(x)");
    }

    /// <summary>
    /// Weryfikuje asymptotyczne zachowanie tanh dla ekstremalnych wartości.
    /// Dla dużych |x|, tanh(x) powinien dążyć do ±1, ale nigdy ich nie osiągnąć.
    /// Ważne dla stabilności numerycznej w przypadku bardzo dużych aktywacji.
    /// </summary>
    [Fact]
    public void Tanh_AsymptoticBehavior()
    {
        // Dla dużych wartości dodatnich
        var largePositive = tanh.Activate(10.0);
        largePositive.ShouldBeGreaterThan(0.99);
        largePositive.ShouldBeLessThan(1.0);
            
        // Dla dużych wartości ujemnych
        var largeNegative = tanh.Activate(-10.0);
        largeNegative.ShouldBeLessThan(-0.99);
        largeNegative.ShouldBeGreaterThan(-1.0);
    }
}
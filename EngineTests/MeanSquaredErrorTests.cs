using Engine.LossFunctions;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Testy dla funkcji straty Mean Squared Error.
/// MSE jest najpopularniejszą funkcją straty dla problemów regresji
/// i fundamentalną dla zrozumienia uczenia maszynowego.
/// </summary>
public class MeanSquaredErrorTests
{
    private readonly MeanSquaredError mse = new MeanSquaredError();
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Sprawdza, że MSE dla perfekcyjnych przewidywań wynosi zero.
    /// Gdy predicted = target, błąd powinien być zero. To jest minimum globalne funkcji straty.
    /// Test weryfikuje, czy funkcja poprawnie identyfikuje sytuację bez błędu.
    /// </summary>
    [Fact]
    public void MSE_ForPerfectPrediction_ShouldReturnZero()
    {
        var predicted = new[] { 1.0, 2.0, 3.0 };
        var target = new[] { 1.0, 2.0, 3.0 };
            
        var loss = mse.CalculateLoss(predicted, target);
        loss.ShouldBe(0.0, Tolerance, "MSE should be zero for perfect predictions");
    }

    /// <summary>
    /// Weryfikuje poprawność obliczania MSE dla znanego przykładu.
    /// Ręcznie obliczamy oczekiwaną wartość i porównujemy z implementacją.
    /// Test sprawdza podstawową arytmetykę: MSE = (1/2n) × Σ(target - prediction)²
    /// </summary>
    [Fact]
    public void MSE_CalculatesCorrectLoss()
    {
        var predicted = new[] { 1.0, 2.0 };
        var target = new[] { 2.0, 1.0 };
            
        // Błędy: (2-1)² + (1-2)² = 1 + 1 = 2
        // MSE = 2 / (2 * 2) = 0.5 (dzielimy przez 2 dla pochodnej i przez length dla średniej)
        const double expectedLoss = 0.5;
        var actualLoss = mse.CalculateLoss(predicted, target);
            
        actualLoss.ShouldBe(expectedLoss, Tolerance, 
            "MSE should be calculated as (1/2n) × Σ(target - prediction)²");
    }

    /// <summary>
    /// Sprawdza, że MSE jest zawsze nieujemne.
    /// MSE jako suma kwadratów nie może być ujemne. To jest matematyczna właściwość,
    /// ale ważna do sprawdzenia w implementacji (mogą być błędy ze znakami).
    /// </summary>
    [Fact]
    public void MSE_IsAlwaysNonNegative()
    {
        var predicted = new[] { -5.0, 0.0, 5.0 };
        var target = new[] { 2.0, -3.0, 1.0 };
            
        var loss = mse.CalculateLoss(predicted, target);
        loss.ShouldBeGreaterThanOrEqualTo(0.0, "MSE should always be non-negative");
    }

    /// <summary>
    /// Weryfikuje poprawność obliczania gradientu MSE dla znanego przykładu.
    /// Gradient MSE = -(target - prediction) / n dla każdego elementu.
    /// To jest punkt startowy dla backpropagation - musi być absolutnie poprawny.
    /// </summary>
    [Fact]
    public void MSE_Derivative_CalculatesCorrectGradient()
    {
        var predicted = new[] { 1.0, 2.0 };
        var target = new[] { 2.0, 1.0 };
            
        // Gradient: -(target - prediction) / length
        // Dla indeksu 0: -(2 - 1) / 2 = -0.5
        // Dla indeksu 1: -(1 - 2) / 2 = 0.5
        var expectedGradient = new[] { -0.5, 0.5 };
        var actualGradient = mse.CalculateDerivative(predicted, target);
            
        actualGradient.Length.ShouldBe(expectedGradient.Length, 
            "Gradient should have same length as input arrays");
            
        for (int i = 0; i < expectedGradient.Length; i++)
        {
            actualGradient[i].ShouldBe(expectedGradient[i], Tolerance,
                $"Gradient at index {i} should be -(target - prediction) / n");
        }
    }

    /// <summary>
    /// Porównuje analityczny gradient MSE z gradientem numerycznym.
    /// To jest najważniejszy test dla funkcji straty - sprawdza, czy gradient jest poprawny.
    /// Używamy perturbacji każdego elementu wektora predicted i obliczamy gradient numerycznie.
    /// Jeśli analityczny gradient się nie zgadza z numerycznym, backpropagation nie zadziała.
    /// </summary>
    [Fact]
    public void MSE_Derivative_MatchesNumericalGradient()
    {
        var predicted = new[] { 1.5, 2.3, -0.7 };
        var target = new[] { 2.0, 1.8, -1.2 };
        const double h = 1e-8;
            
        var analyticalGradient = mse.CalculateDerivative(predicted, target);
            
        // Sprawdzamy gradient numerycznie dla każdego elementu
        for (int i = 0; i < predicted.Length; i++)
        {
            // Tworzymy kopie z małą perturbacją
            var predictedPlus = (double[])predicted.Clone();
            var predictedMinus = (double[])predicted.Clone();
                
            predictedPlus[i] += h;
            predictedMinus[i] -= h;
                
            // Gradient numeryczny
            var lossPlus = mse.CalculateLoss(predictedPlus, target);
            var lossMinus = mse.CalculateLoss(predictedMinus, target);
            var numericalGradient = (lossPlus - lossMinus) / (2 * h);
                
            analyticalGradient[i].ShouldBe(numericalGradient, 1e-6,
                $"Analytical gradient should match numerical gradient at index {i}");
        }
    }

    /// <summary>
    /// Sprawdza, że funkcja rzuca wyjątek dla niepasujących rozmiarów tablic.
    /// Robustność kodu - musi gracefully handle błędne input data.
    /// W praktyce różne rozmiary predicted i target oznaczają błąd w architekturze sieci.
    /// </summary>
    [Fact]
    public void MSE_ThrowsException_ForMismatchedArrayLengths()
    {
        var predicted = new[] { 1.0, 2.0 };
        var target = new[] { 1.0, 2.0, 3.0 }; // Różna długość!
            
        Should.Throw<ArgumentException>(() => mse.CalculateLoss(predicted, target))
            .Message.ShouldContain("must have the same length");
    }

    /// <summary>
    /// Weryfikuje, że MSE jest funkcją kwadratową względem błędu.
    /// Podwojenie błędu powinno spowodować poczwórne zwiększenie straty.
    /// Ta właściwość jest ważna dla zrozumienia dynamics of learning.
    /// </summary>
    [Fact]
    public void MSE_IsQuadraticInError()
    {
        var target = new[] { 1.0 };
            
        var predicted1 = new[] { 2.0 }; // błąd = 1
        var predicted2 = new[] { 3.0 }; // błąd = 2
            
        var loss1 = mse.CalculateLoss(predicted1, target);
        var loss2 = mse.CalculateLoss(predicted2, target);
            
        // loss2 powinno być ~4 razy większe niż loss1
        var ratio = loss2 / loss1;
        ratio.ShouldBe(4.0, 0.1, "MSE should be quadratic in error magnitude");
    }

    /// <summary>
    /// Sprawdza, że gradient MSE ma przeciwny kierunek do błędu.
    /// Jeśli prediction > target, gradient powinien być dodatni (żeby zmniejszyć prediction).
    /// Jeśli prediction < target, gradient powinien być ujemny (żeby zwiększyć prediction).
    /// Ta właściwość zapewnia, że uczenie idzie w kierunku minimalizacji błędu.
    /// </summary>
    [Fact]
    public void MSE_Gradient_PointsTowardsCorrection()
    {
        var target = new[] { 1.0, 1.0 };
        var predicted = new[] { 2.0, 0.5 }; // pierwsza za wysoka, druga za niska
            
        var gradient = mse.CalculateDerivative(predicted, target);
            
        // Gradient dla pierwszego elementu powinien być dodatni (żeby zmniejszyć prediction)
        gradient[0].ShouldBeGreaterThan(0, 
            "Gradient should be positive when prediction > target");
        // Gradient dla drugiego elementu powinien być ujemny (żeby zwiększyć prediction)  
        gradient[1].ShouldBeLessThan(0, 
            "Gradient should be negative when prediction < target");
    }
}
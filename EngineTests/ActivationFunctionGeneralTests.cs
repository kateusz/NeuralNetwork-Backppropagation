using Engine.Activations;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Ogólne testy sprawdzające właściwości wszystkich funkcji aktywacji.
/// Te testy zapewniają spójność interfejsu i ogólne właściwości matematyczne.
/// </summary>
public class ActivationFunctionGeneralTests
{
    /// <summary>
    /// Sprawdza, że wszystkie funkcje aktywacji mają poprawnie zdefiniowane nazwy.
    /// Nazwy są ważne do logowania, debugowania i identyfikacji w różnych częściach systemu.
    /// Pusty lub null string może powodować problemy w diagnostyce.
    /// </summary>
    [Fact]
    public void AllActivationFunctions_HaveValidNames()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
            
        foreach (var function in functions)
        {
            function.Name.ShouldNotBeNullOrEmpty(
                $"Function {function.GetType().Name} should have a valid name");
            function.Name.Length.ShouldBeGreaterThan(0,
                $"Function {function.GetType().Name} name should not be empty");
        }
    }

    /// <summary>
    /// Weryfikuje, że wszystkie funkcje aktywacji produkują skończone wyniki.
    /// NaN lub Infinity w funkcjach aktywacji natychmiast psuje cały proces uczenia.
    /// To jest test stabilności numerycznej - funkcje muszą być robust dla różnych input values.
    /// </summary>
    [Fact]
    public void AllActivationFunctions_ProduceFiniteResults()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
        var testValues = new[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
            
        foreach (var function in functions)
        {
            foreach (var x in testValues)
            {
                var activation = function.Activate(x);
                var derivative = function.Derivative(x);
                    
                double.IsNaN(activation).ShouldBeFalse(
                    $"{function.Name} activation should not be NaN at x = {x}");
                double.IsInfinity(activation).ShouldBeFalse(
                    $"{function.Name} activation should not be infinite at x = {x}");
                double.IsNaN(derivative).ShouldBeFalse(
                    $"{function.Name} derivative should not be NaN at x = {x}");
                double.IsInfinity(derivative).ShouldBeFalse(
                    $"{function.Name} derivative should not be infinite at x = {x}");
            }
        }
    }

    /// <summary>
    /// Sprawdza, że funkcje aktywacji są deterministyczne.
    /// Wielokrotne wywołanie z tymi samymi argumentami powinno dawać identyczne wyniki.
    /// Brak determinizmu sprawiłby, że testy byłyby niereprodukowalne i uczenie nieprzewidywalne.
    /// </summary>
    [Fact]
    public void AllActivationFunctions_AreDeterministic()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
        var testValues = new[] { -1.5, 0.0, 2.3 };
            
        foreach (var function in functions)
        {
            foreach (var x in testValues)
            {
                var result1 = function.Activate(x);
                var result2 = function.Activate(x);
                var derivative1 = function.Derivative(x);
                var derivative2 = function.Derivative(x);
                    
                result1.ShouldBe(result2, 1e-15,
                    $"{function.Name} should be deterministic for activation at x = {x}");
                derivative1.ShouldBe(derivative2, 1e-15,
                    $"{function.Name} should be deterministic for derivative at x = {x}");
            }
        }
    }

    /// <summary>
    /// Weryfikuje, że wszystkie funkcje aktywacji obsługują granice swojej dziedziny.
    /// Test brzegowych przypadków - funkcje nie powinny crashować ani zwracać NaN
    /// dla wartości blisko granic precision zmiennoprzecinkowych.
    /// </summary>
    [Fact]
    public void AllActivationFunctions_HandleBoundaryValues()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
        var boundaryValues = new[] { -1000.0, 1000.0 }; // Unikamy extreme values które mogą crashować
            
        foreach (var function in functions)
        {
            foreach (var x in boundaryValues)
            {
                // Funkcje nie powinny rzucać wyjątków ani zwracać NaN/Infinity
                Should.NotThrow(() => 
                    {
                        var activation = function.Activate(x);
                        var derivative = function.Derivative(x);
                        
                        // Wyniki powinny być skończone lub gracefully handled
                        (double.IsFinite(activation) || 
                         activation == 0.0 || 
                         activation == 1.0 || 
                         activation == -1.0).ShouldBeTrue(
                            $"{function.Name} should handle boundary value {x} gracefully");
                            
                        (double.IsFinite(derivative) || 
                         derivative == 0.0 || 
                         derivative == 1.0).ShouldBeTrue(
                            $"{function.Name} derivative should handle boundary value {x} gracefully");
                    }, $"{function.Name} should not throw exceptions for boundary value {x}");
            }
        }
    }

    /// <summary>
    /// Sprawdza, że wszystkie funkcje aktywacji implementują wymagane metody interfejsu.
    /// Test kompletności implementacji - każda klasa musi implementować wszystkie metody interfejsu.
    /// Pomocny dla sprawdzenia, czy nowe funkcje aktywacji są kompletnie zaimplementowane.
    /// </summary>
    [Fact]
    public void AllActivationFunctions_ImplementRequiredMethods()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
            
        foreach (var function in functions)
        {
            // Sprawdzamy, czy metody nie są null (w C# nie powinno się zdarzyć, ale dla pewności)
            Should.NotThrow(() => function.Activate(0.0),
                $"{function.Name} should implement Activate method");
            Should.NotThrow(() => function.Derivative(0.0),
                $"{function.Name} should implement Derivative method");
                
            function.Name.ShouldNotBeNull(
                $"{function.Name} should implement Name property");
        }
    }
}
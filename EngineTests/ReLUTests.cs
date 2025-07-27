using Engine.Activations;
using Shouldly;

namespace EngineTests
{
    /// <summary>
    /// Testy dla funkcji aktywacji ReLU (Rectified Linear Unit).
    /// ReLU jest obecnie najczęściej używaną funkcją aktywacji w głębokich sieciach neuronowych
    /// ze względu na prostotę i brak problemu vanishing gradient.
    /// </summary>
    public class ReLUTests
    {
        private readonly ReLU relu = new ReLU();

        /// <summary>
        /// Weryfikuje, że ReLU dla wartości dodatnich zwraca wejście bez zmian.
        /// To jest definicyjna właściwość ReLU: f(x) = x dla x > 0.
        /// Ważne, bo pokazuje liniowość ReLU w dodatniej części dziedziny.
        /// </summary>
        [Fact]
        public void ReLU_ForPositiveValues_ReturnsInput()
        {
            var testValues = new[] { 0.1, 1.0, 5.0, 100.0 };
            
            foreach (var x in testValues)
            {
                relu.Activate(x).ShouldBe(x, 
                    $"ReLU should return input unchanged for positive value {x}");
            }
        }

        /// <summary>
        /// Sprawdza, że ReLU dla wartości ujemnych zawsze zwraca zero.
        /// To jest druga część definicji ReLU: f(x) = 0 dla x < 0.
        /// Ta właściwość powoduje "wyłączanie" niektórych neuronów, co pomaga w regularyzacji.
        /// </summary>
        [Fact]
        public void ReLU_ForNegativeValues_ReturnsZero()
        {
            var testValues = new[] { -0.1, -1.0, -5.0, -100.0 };
            
            foreach (var x in testValues)
            {
                relu.Activate(x).ShouldBe(0.0, 
                    $"ReLU should return zero for negative value {x}");
            }
        }

        /// <summary>
        /// Testuje zachowanie ReLU dokładnie w punkcie zero.
        /// Matematycznie: f(0) = max(0, 0) = 0.
        /// To jest przypadek brzegowy, ale ważny dla completeness of definition.
        /// </summary>
        [Fact]
        public void ReLU_AtZero_ReturnsZero()
        {
            relu.Activate(0.0).ShouldBe(0.0);
        }

        /// <summary>
        /// Weryfikuje, że pochodna ReLU dla wartości dodatnich wynosi 1.
        /// f'(x) = 1 dla x > 0. To oznacza, że ReLU nie zmienia gradientu w dodatniej części,
        /// co rozwiązuje problem vanishing gradient znaný z sigmoid i tanh.
        /// </summary>
        [Fact]
        public void ReLU_Derivative_ForPositiveValues_ReturnsOne()
        {
            var testValues = new[] { 0.1, 1.0, 5.0, 100.0 };
            
            foreach (var x in testValues)
            {
                relu.Derivative(x).ShouldBe(1.0, 
                    $"ReLU derivative should be 1 for positive value {x}");
            }
        }

        /// <summary>
        /// Sprawdza, że pochodna ReLU dla wartości ujemnych wynosi zero.
        /// f'(x) = 0 dla x < 0. To powoduje, że neurony z ujemną aktywacją nie uczą się
        /// (ich gradienty są zerowe), co może być problemem zwanym "dying ReLU".
        /// </summary>
        [Fact]
        public void ReLU_Derivative_ForNegativeValues_ReturnsZero()
        {
            var testValues = new[] { -0.1, -1.0, -5.0, -100.0 };
            
            foreach (var x in testValues)
            {
                relu.Derivative(x).ShouldBe(0.0, 
                    $"ReLU derivative should be 0 for negative value {x}");
            }
        }

        /// <summary>
        /// Testuje konwencję dla pochodnej ReLU w punkcie zero.
        /// Matematycznie pochodna nie istnieje w x=0, ale konwencjonalnie przyjmujemy f'(0) = 0.
        /// Ta konwencja jest arbitralna, ale musi być konsystentna w całej implementacji.
        /// </summary>
        [Fact]
        public void ReLU_Derivative_AtZero_ReturnsZero()
        {
            relu.Derivative(0.0).ShouldBe(0.0, 
                "ReLU derivative at zero should be 0 by convention");
        }

        /// <summary>
        /// Weryfikuje, że ReLU jest funkcją niemalejącą (monotonically non-decreasing).
        /// Dla x1 ≤ x2 powinno być ReLU(x1) ≤ ReLU(x2).
        /// Ta właściwość jest ważna dla zachowania porządku w danych.
        /// </summary>
        [Fact]
        public void ReLU_IsMonotonicallyNonDecreasing()
        {
            var testValues = new[] { -5.0, -1.0, 0.0, 1.0, 5.0 };
            
            for (int i = 0; i < testValues.Length - 1; i++)
            {
                var y1 = relu.Activate(testValues[i]);
                var y2 = relu.Activate(testValues[i + 1]);
                y2.ShouldBeGreaterThanOrEqualTo(y1, 
                    $"ReLU should be non-decreasing: f({testValues[i + 1]}) should be ≥ f({testValues[i]})");
            }
        }
    }
}
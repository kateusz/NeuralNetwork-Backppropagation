using Engine.Core;
using Engine.Activations;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Comprehensive tests dla Layer class.
/// Sprawdzają poprawność matematyczną, integrację z activation functions,
/// oraz edge cases. Każdy test weryfikuje konkretny aspekt forward propagation.
/// </summary>
public class LayerTests
{
    private const double Tolerance = 1e-10;

    #region Constructor Tests

    /// <summary>
    /// Weryfikuje, że konstruktor poprawnie ustawia podstawowe właściwości warstwy.
    /// </summary>
    [Fact]
    public void Layer_Constructor_SetsBasicProperties()
    {
        var activation = new Sigmoid();
        var layer = new Layer(3, 2, activation);

        layer.InputSize.ShouldBe(3);
        layer.OutputSize.ShouldBe(2);
        layer.ActivationName.ShouldBe("Sigmoid");
    }

    /// <summary>
    /// Sprawdza walidację parametrów konstruktora - negative/zero sizes powinny rzucać wyjątki.
    /// </summary>
    [Theory]
    [InlineData(0, 1)]
    [InlineData(-1, 1)]
    [InlineData(1, 0)]
    [InlineData(1, -1)]
    public void Layer_Constructor_ThrowsException_ForInvalidSizes(int inputSize, int outputSize)
    {
        var activation = new Sigmoid();
        
        Should.Throw<ArgumentException>(() => new Layer(inputSize, outputSize, activation));
    }

    /// <summary>
    /// Weryfikuje, że konstruktor rzuca wyjątek dla null activation function.
    /// </summary>
    [Fact]
    public void Layer_Constructor_ThrowsException_ForNullActivation()
    {
        Should.Throw<ArgumentNullException>(() => new Layer(2, 3, null!));
    }

    #endregion

    #region Forward Pass Mathematical Correctness

    /// <summary>
    /// Testuje Forward pass z hand-calculated example.
    /// Używa known weights i biases dla pełnej kontroli nad wynikami.
    /// </summary>
    [Fact]
    public void Layer_Forward_ComputesCorrectOutput_WithKnownWeights()
    {
        // Setup: 2 inputs → 3 outputs z ReLU activation
        var layer = new Layer(2, 3, new ReLU());
        
        // Ustawiamy znane wagi i biasy
        var weights = new Matrix(new double[,]
        {
            { 0.5, 0.3 }, // Neuron 1: w11=0.5, w12=0.3
            { 0.2, 0.8 }, // Neuron 2: w21=0.2, w22=0.8
            { 0.1, 0.6 } // Neuron 3: w31=0.1, w32=0.6
        });
        var biases = new double[] { 0.1, 0.2, 0.3 };
        
        layer.SetWeights(weights);
        layer.SetBiases(biases);
        
        // Test input
        var inputs = new double[] { 2.0, 1.5 };
        
        // Forward pass
        var outputs = layer.Forward(inputs);
        
        // Hand-calculated expected results:
        // Neuron 1: z1 = 0.5×2.0 + 0.3×1.5 + 0.1 = 1.0 + 0.45 + 0.1 = 1.55, a1 = ReLU(1.55) = 1.55
        // Neuron 2: z2 = 0.2×2.0 + 0.8×1.5 + 0.2 = 0.4 + 1.2 + 0.2 = 1.8, a2 = ReLU(1.8) = 1.8
        // Neuron 3: z3 = 0.1×2.0 + 0.6×1.5 + 0.3 = 0.2 + 0.9 + 0.3 = 1.4, a3 = ReLU(1.4) = 1.4
        
        outputs.Length.ShouldBe(3);
        outputs[0].ShouldBe(1.55, Tolerance, "Neuron 1 output incorrect");
        outputs[1].ShouldBe(1.8, Tolerance, "Neuron 2 output incorrect");
        outputs[2].ShouldBe(1.4, Tolerance, "Neuron 3 output incorrect");
    }

    /// <summary>
    /// Testuje Forward pass z negative net inputs i ReLU activation.
    /// Sprawdza czy ReLU poprawnie zeruje negative values.
    /// </summary>
    [Fact]
    public void Layer_Forward_HandlesNegativeNetInputs_WithReLU()
    {
        var layer = new Layer(2, 2, new ReLU());
        
        var weights = new Matrix(new double[,] 
        {
            { -1.0, 0.5 },  // Neuron 1: może dać negative result
            { 0.3, -0.8 }   // Neuron 2: może dać negative result
        });
        var biases = new double[] { 0.2, -0.1 };
        
        layer.SetWeights(weights);
        layer.SetBiases(biases);
        
        var inputs = new double[] { 0.5, 0.3 };
        var outputs = layer.Forward(inputs);
        
        // Hand calculations:
        // Neuron 1: z1 = -1.0×0.5 + 0.5×0.3 + 0.2 = -0.5 + 0.15 + 0.2 = -0.15, a1 = ReLU(-0.15) = 0.0
        // Neuron 2: z2 = 0.3×0.5 + (-0.8)×0.3 + (-0.1) = 0.15 - 0.24 - 0.1 = -0.19, a2 = ReLU(-0.19) = 0.0
        
        outputs[0].ShouldBe(0.0, Tolerance, "Negative net input should be zeroed by ReLU");
        outputs[1].ShouldBe(0.0, Tolerance, "Negative net input should be zeroed by ReLU");
    }

    /// <summary>
    /// Weryfikuje Forward pass z Sigmoid activation i known mathematical properties.
    /// </summary>
    [Fact]
    public void Layer_Forward_WithSigmoid_ProducesValidProbabilities()
    {
        var layer = new Layer(1, 3, new Sigmoid());
        
        var weights = new Matrix(new double[,] 
        {
            { 10.0 },   // Large positive weight
            { 0.0 },    // Zero weight  
            { -10.0 }   // Large negative weight
        });
        var biases = new double[] { 0.0, 0.0, 0.0 };
        
        layer.SetWeights(weights);
        layer.SetBiases(biases);
        
        var inputs = new double[] { 1.0 };
        var outputs = layer.Forward(inputs);
        
        // Expected behavior:
        // Neuron 1: sigmoid(10.0) → close to 1.0
        // Neuron 2: sigmoid(0.0) = 0.5  
        // Neuron 3: sigmoid(-10.0) → close to 0.0
        
        outputs[0].ShouldBeGreaterThan(0.99, "Large positive input should give sigmoid ≈ 1");
        outputs[1].ShouldBe(0.5, Tolerance, "Zero input should give sigmoid = 0.5");
        outputs[2].ShouldBeLessThan(0.01, "Large negative input should give sigmoid ≈ 0");
        
        // All sigmoid outputs should be in [0,1]
        outputs.ShouldAllBe(x => x >= 0.0 && x <= 1.0, "Sigmoid outputs should be probabilities");
    }

    #endregion

    #region Integration with Activation Functions

    /// <summary>
    /// Testuje integrację ze wszystkimi existing activation functions.
    /// Każda powinna działać bez błędów i produkować sensowne wyniki.
    /// </summary>
    [Fact]
    public void Layer_Forward_WorksWithAllActivationFunctions()
    {
        IActivationFunction[] functions = { new Sigmoid(), new ReLU(), new Tanh() };
        var inputs = new double[] { 1.0, -0.5, 2.0 };
        
        foreach (var activation in functions)
        {
            var layer = new Layer(3, 2, activation);
            
            // Forward pass nie powinien rzucać wyjątków
            Should.NotThrow(() => layer.Forward(inputs), 
                $"Forward pass should work with {activation.Name}");
            
            var outputs = layer.Forward(inputs);
            
            // Outputs powinny być finite
            outputs.ShouldAllBe(x => double.IsFinite(x), 
                $"All outputs should be finite for {activation.Name}");
            
            // Outputs powinny mieć correct size
            outputs.Length.ShouldBe(2, 
                $"Output size should match layer output size for {activation.Name}");
        }
    }

    /// <summary>
    /// Weryfikuje, że różne activation functions dają różne wyniki dla tych samych inputs.
    /// (z wyjątkiem edge cases gdzie mogą się pokrywać)
    /// </summary>
    [Fact]
    public void Layer_Forward_DifferentActivations_ProduceDifferentOutputs()
    {
        var inputs = new double[] { 1.0, 0.5 };
        
        var sigmoidLayer = new Layer(2, 1, new Sigmoid());
        var reluLayer = new Layer(2, 1, new ReLU());
        var tanhLayer = new Layer(2, 1, new Tanh());
        
        // Użyj tych samych wag dla fair comparison
        var weights = new Matrix(new double[,] { { 0.5, 0.3 } });
        var biases = new double[] { 0.2 };
        
        sigmoidLayer.SetWeights(weights);
        sigmoidLayer.SetBiases(biases);
        reluLayer.SetWeights(weights);
        reluLayer.SetBiases(biases);
        tanhLayer.SetWeights(weights);
        tanhLayer.SetBiases(biases);
        
        var sigmoidOutput = sigmoidLayer.Forward(inputs)[0];
        var reluOutput = reluLayer.Forward(inputs)[0];
        var tanhOutput = tanhLayer.Forward(inputs)[0];
        
        // Net input = 0.5×1.0 + 0.3×0.5 + 0.2 = 0.85
        // sigmoid(0.85) ≈ 0.70, ReLU(0.85) = 0.85, tanh(0.85) ≈ 0.69
        
        Math.Abs(sigmoidOutput - reluOutput).ShouldBeGreaterThan(1e-6, "Sigmoid and ReLU should produce different outputs");
        Math.Abs(sigmoidOutput - tanhOutput).ShouldBeGreaterThan(1e-6, "Sigmoid and Tanh should produce different outputs");
        Math.Abs(reluOutput - tanhOutput).ShouldBeGreaterThan(1e-6, "ReLU and Tanh should produce different outputs");
    }

    #endregion

    #region Error Handling and Validation

    /// <summary>
    /// Sprawdza walidację rozmiaru input vector w Forward method.
    /// </summary>
    [Fact]
    public void Layer_Forward_ThrowsException_ForWrongInputSize()
    {
        var layer = new Layer(3, 2, new ReLU());
        
        // Test z wrong sizes
        var tooSmall = new double[] { 1.0, 2.0 };        // Expected 3, got 2
        var tooBig = new double[] { 1.0, 2.0, 3.0, 4.0 }; // Expected 3, got 4
        
        Should.Throw<ArgumentException>(() => layer.Forward(tooSmall))
            .Message.ShouldContain("doesn't match layer input size");
        
        Should.Throw<ArgumentException>(() => layer.Forward(tooBig))
            .Message.ShouldContain("doesn't match layer input size");
    }

    /// <summary>
    /// Weryfikuje handling null inputs.
    /// </summary>
    [Fact]
    public void Layer_Forward_ThrowsException_ForNullInput()
    {
        var layer = new Layer(2, 1, new Sigmoid());
        
        Should.Throw<ArgumentNullException>(() => layer.Forward(null!));
    }

    /// <summary>
    /// Testuje SetWeights method validation.
    /// </summary>
    [Fact]
    public void Layer_SetWeights_ValidatesWeightMatrixSize()
    {
        var layer = new Layer(2, 3, new ReLU());
        
        // Correct size should work
        var correctWeights = new Matrix(new double[3, 2]);
        Should.NotThrow(() => layer.SetWeights(correctWeights));
        
        // Wrong sizes should throw
        var wrongSize1 = new Matrix(new double[2, 2]); // Wrong output dimension
        var wrongSize2 = new Matrix(new double[3, 3]); // Wrong input dimension
        
        Should.Throw<ArgumentException>(() => layer.SetWeights(wrongSize1));
        Should.Throw<ArgumentException>(() => layer.SetWeights(wrongSize2));
        Should.Throw<ArgumentNullException>(() => layer.SetWeights((Matrix)null!));
    }

    /// <summary>
    /// Testuje SetBiases method validation.
    /// </summary>
    [Fact]
    public void Layer_SetBiases_ValidatesBiasVectorSize()
    {
        var layer = new Layer(2, 3, new ReLU());
        
        // Correct size should work
        var correctBiases = new double[3];
        Should.NotThrow(() => layer.SetBiases(correctBiases));
        
        // Wrong sizes should throw
        var wrongSize = new double[2];
        
        Should.Throw<ArgumentException>(() => layer.SetBiases(wrongSize));
        Should.Throw<ArgumentNullException>(() => layer.SetBiases(null!));
    }

    #endregion

    #region Caching Behavior

    /// <summary>
    /// Weryfikuje, że Layer poprawnie cache'uje intermediate values podczas Forward pass.
    /// Te wartości są kluczowe dla późniejszego backward pass.
    /// </summary>
    [Fact]
    public void Layer_Forward_CachesIntermediateValues()
    {
        var layer = new Layer(2, 2, new Sigmoid());
        
        // Przed Forward pass - brak cached values
        layer.HasCachedValues.ShouldBeFalse();
        layer.LastInputs.ShouldBeNull();
        layer.LastNetInputs.ShouldBeNull();
        layer.LastOutputs.ShouldBeNull();
        
        var inputs = new double[] { 1.0, 0.5 };
        var outputs = layer.Forward(inputs);
        
        // Po Forward pass - powinny być cached values
        layer.HasCachedValues.ShouldBeTrue();
        
        var cachedInputs = layer.LastInputs;
        var cachedNetInputs = layer.LastNetInputs;
        var cachedOutputs = layer.LastOutputs;
        
        // Cached inputs powinny być identical z original inputs
        cachedInputs.ShouldNotBeNull();
        cachedInputs!.Length.ShouldBe(2);
        cachedInputs[0].ShouldBe(inputs[0], Tolerance);
        cachedInputs[1].ShouldBe(inputs[1], Tolerance);
        
        // Cached outputs powinny być identical z returned outputs
        cachedOutputs.ShouldNotBeNull();
        cachedOutputs!.Length.ShouldBe(2);
        cachedOutputs[0].ShouldBe(outputs[0], Tolerance);
        cachedOutputs[1].ShouldBe(outputs[1], Tolerance);
        
        // Cached net inputs powinny być different od outputs (przed aktywacją)
        cachedNetInputs.ShouldNotBeNull();
        cachedNetInputs!.Length.ShouldBe(2);
        // Net inputs != outputs dla non-identity activation
        cachedNetInputs.ShouldNotBe(cachedOutputs);
    }

    /// <summary>
    /// Sprawdza, że cached values są independent copies (defensive copying).
    /// Modyfikacja original arrays nie powinna wpływać na cached values.
    /// </summary>
    [Fact]
    public void Layer_Forward_CachesDefensiveCopies()
    {
        var layer = new Layer(2, 1, new ReLU());
        
        var inputs = new double[] { 1.0, 2.0 };
        layer.Forward(inputs);
        
        // Modyfikuj original input array
        inputs[0] = 999.0;
        inputs[1] = 888.0;
        
        // Cached values powinny remain unchanged
        var cachedInputs = layer.LastInputs!;
        cachedInputs[0].ShouldBe(1.0, Tolerance, "Cached inputs should be defensive copies");
        cachedInputs[1].ShouldBe(2.0, Tolerance, "Cached inputs should be defensive copies");
    }

    #endregion

    #region Weight and Bias Management

    /// <summary>
    /// Testuje roundtrip: SetWeights → GetWeights.
    /// </summary>
    [Fact]
    public void Layer_WeightManagement_Roundtrip()
    {
        var layer = new Layer(2, 3, new ReLU());
        
        var originalWeights = new Matrix(new double[,] 
        {
            { 0.1, 0.2 },
            { 0.3, 0.4 },
            { 0.5, 0.6 }
        });
        
        layer.SetWeights(originalWeights);
        var retrievedWeights = layer.GetWeightsMatrix();
        
        // Retrieved weights powinny match original
        retrievedWeights.Rows.ShouldBe(3);
        retrievedWeights.Cols.ShouldBe(2);
        
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                retrievedWeights[i, j].ShouldBe(originalWeights[i, j], Tolerance,
                    $"Weight [{i},{j}] should match");
            }
        }
        
        // Retrieved weights powinny być copy, nie reference
        originalWeights[0, 0] = 999.0;
        retrievedWeights[0, 0].ShouldNotBe(999.0, "Retrieved weights should be a copy");
    }

    /// <summary>
    /// Testuje roundtrip: SetBiases → GetBiases.
    /// </summary>
    [Fact]
    public void Layer_BiasManagement_Roundtrip()
    {
        var layer = new Layer(2, 3, new ReLU());
        
        var originalBiases = new double[] { 0.1, 0.2, 0.3 };
        
        layer.SetBiases(originalBiases);
        var retrievedBiases = layer.GetBiases();
        
        retrievedBiases.Length.ShouldBe(3);
        for (int i = 0; i < 3; i++)
        {
            retrievedBiases[i].ShouldBe(originalBiases[i], Tolerance,
                $"Bias [{i}] should match");
        }
        
        // Retrieved biases powinny być copy, nie reference
        originalBiases[0] = 999.0;
        retrievedBiases[0].ShouldNotBe(999.0, "Retrieved biases should be a copy");
    }

    /// <summary>
    /// Weryfikuje, że random weight initialization produkuje reasonable values.
    /// </summary>
    [Fact]
    public void Layer_RandomInitialization_ProducesReasonableWeights()
    {
        var layer = new Layer(10, 5, new ReLU());
        
        var weights = layer.GetWeightsMatrix();
        var biases = layer.GetBiases();
        
        // Weights powinny być small (dla Xavier initialization)
        var weightValues = new List<double>();
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Cols; j++)
            {
                weightValues.Add(weights[i, j]);
            }
        }
        
        // Większość wag powinna być w reasonable range
        var absWeights = weightValues.Select(Math.Abs);
        absWeights.Max().ShouldBeLessThan(2.0, "Weights should not be too large");
        absWeights.Average().ShouldBeLessThan(1.0, "Average weight magnitude should be reasonable");
        
        // Biases powinny być zero
        biases.ShouldAllBe(x => Math.Abs(x) < 1e-10, "Biases should be initialized to zero");
        
        // Weights powinny być varied (not all the same)
        var distinctWeights = weightValues.Distinct().Count();
        distinctWeights.ShouldBeGreaterThan(weightValues.Count / 2, 
            "Weights should be varied, not all the same");
    }

    #endregion

    #region Performance and Edge Cases

    /// <summary>
    /// Testuje determinism - multiple calls z tymi samymi inputs dają identical results.
    /// </summary>
    [Fact]
    public void Layer_Forward_IsDeterministic()
    {
        var layer = new Layer(3, 2, new Tanh());
        var inputs = new double[] { 1.0, -0.5, 2.0 };
        
        var result1 = layer.Forward(inputs);
        var result2 = layer.Forward(inputs);
        var result3 = layer.Forward(inputs);
        
        // All results should be identical
        for (int i = 0; i < result1.Length; i++)
        {
            result1[i].ShouldBe(result2[i], Tolerance, $"Results should be deterministic at index {i}");
            result1[i].ShouldBe(result3[i], Tolerance, $"Results should be deterministic at index {i}");
        }
    }

    /// <summary>
    /// Testuje ToString method dla debugging purposes.
    /// </summary>
    [Fact]
    public void Layer_ToString_ProvidesUsefulInformation()
    {
        var layer = new Layer(784, 128, new ReLU());
        var description = layer.ToString();
        
        description.ShouldContain("784", Case.Insensitive, "Should contain input size");
        description.ShouldContain("128", Case.Insensitive, "Should contain output size");
        description.ShouldContain("ReLU", Case.Insensitive, "Should contain activation function name");
    }

    #endregion
}
using Engine.Core;
using Shouldly;

namespace EngineTests;

/// <summary>
/// Comprehensive tests dla Matrix class.
/// Weryfikują mathematical correctness, SIMD optimizations, error handling i edge cases.
/// Matrix jest critical component dla neural networks, więc testing musi być thorough.
/// </summary>
public class MatrixTests
{
    private const double Tolerance = 1e-10;

    #region Constructor Tests

    /// <summary>
    /// Weryfikuje, że podstawowy konstruktor tworzy matrix wypełnioną zerami.
    /// </summary>
    [Fact]
    public void Matrix_BasicConstructor_CreatesZeroMatrix()
    {
        var matrix = new Matrix(3, 4);

        matrix.Rows.ShouldBe(3);
        matrix.Cols.ShouldBe(4);
        matrix.Size.ShouldBe(12);
        matrix.IsSquare.ShouldBeFalse();
        matrix.IsVector.ShouldBeFalse();

        // Wszystkie elementy powinny być zero
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                matrix[i, j].ShouldBe(0.0, Tolerance, $"Element [{i},{j}] should be zero");
            }
        }
    }

    /// <summary>
    /// Testuje konstruktor z 2D array i weryfikuje correct data copying.
    /// </summary>
    [Fact]
    public void Matrix_Array2DConstructor_CopiesDataCorrectly()
    {
        var sourceData = new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        };

        var matrix = new Matrix(sourceData);

        matrix.Rows.ShouldBe(2);
        matrix.Cols.ShouldBe(3);
        matrix[0, 0].ShouldBe(1.0, Tolerance);
        matrix[0, 1].ShouldBe(2.0, Tolerance);
        matrix[0, 2].ShouldBe(3.0, Tolerance);
        matrix[1, 0].ShouldBe(4.0, Tolerance);
        matrix[1, 1].ShouldBe(5.0, Tolerance);
        matrix[1, 2].ShouldBe(6.0, Tolerance);

        // Verify defensive copying - modifying source shouldn't affect matrix
        sourceData[0, 0] = 999.0;
        matrix[0, 0].ShouldBe(1.0, Tolerance, "Matrix should be defensive copy");
    }

    /// <summary>
    /// Testuje konstruktor z 1D array w row-major format.
    /// </summary>
    [Fact]
    public void Matrix_Array1DConstructor_InterpetsRowMajorCorrectly()
    {
        var data = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        var matrix = new Matrix(data, 2, 3);

        matrix.Rows.ShouldBe(2);
        matrix.Cols.ShouldBe(3);
        
        // Row-major interpretation: [1,2,3] [4,5,6]
        matrix[0, 0].ShouldBe(1.0, Tolerance);
        matrix[0, 1].ShouldBe(2.0, Tolerance);
        matrix[0, 2].ShouldBe(3.0, Tolerance);
        matrix[1, 0].ShouldBe(4.0, Tolerance);
        matrix[1, 1].ShouldBe(5.0, Tolerance);
        matrix[1, 2].ShouldBe(6.0, Tolerance);
    }

    /// <summary>
    /// Sprawdza validation w konstruktorach.
    /// </summary>
    [Theory]
    [InlineData(0, 1)]
    [InlineData(-1, 1)]
    [InlineData(1, 0)]
    [InlineData(1, -1)]
    public void Matrix_Constructor_ValidatesDimensions(int rows, int cols)
    {
        Should.Throw<ArgumentException>(() => new Matrix(rows, cols));
    }

    /// <summary>
    /// Sprawdza validation w konstruktorze z 1D array.
    /// </summary>
    [Fact]
    public void Matrix_Array1DConstructor_ValidatesDataSize()
    {
        var data = new double[] { 1.0, 2.0, 3.0 };
        
        // Data length (3) doesn't match dimensions (2×3=6)
        Should.Throw<ArgumentException>(() => new Matrix(data, 2, 3))
            .Message.ShouldContain("doesn't match matrix size");
    }

    #endregion

    #region Matrix Properties Tests

    /// <summary>
    /// Testuje property calculations dla różnych matrix shapes.
    /// </summary>
    [Theory]
    [InlineData(3, 3, true, false)]   // Square matrix
    [InlineData(1, 5, false, true)]   // Row vector  
    [InlineData(5, 1, false, true)]   // Column vector
    [InlineData(3, 4, false, false)]  // General matrix
    public void Matrix_Properties_CalculatedCorrectly(int rows, int cols, bool expectedSquare, bool expectedVector)
    {
        var matrix = new Matrix(rows, cols);

        matrix.IsSquare.ShouldBe(expectedSquare);
        matrix.IsVector.ShouldBe(expectedVector);
        matrix.Size.ShouldBe(rows * cols);
    }

    #endregion

    #region Matrix-Vector Multiplication Tests

    /// <summary>
    /// Testuje matrix-vector multiplication z hand-calculated example.
    /// To jest najważniejsza operacja w neural networks (forward propagation).
    /// </summary>
    [Fact]
    public void Matrix_MultiplyVector_ComputesCorrectResult()
    {
        // Setup: 2×3 matrix × 3×1 vector = 2×1 result
        var matrixData = new double[,]
        {
            { 1.0, 2.0, 3.0 },  // Row 1: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            { 4.0, 5.0, 6.0 }   // Row 2: 4*4 + 5*5 + 6*6 = 16 + 25 + 36 = 77
        };
        var matrix = new Matrix(matrixData);
        var vector = new double[] { 4.0, 5.0, 6.0 };

        var result = matrix.MultiplyVector(vector);

        result.Length.ShouldBe(2);
        result[0].ShouldBe(32.0, Tolerance, "First element incorrect");
        result[1].ShouldBe(77.0, Tolerance, "Second element incorrect");
    }

    /// <summary>
    /// Weryfikuje, że matrix-vector multiplication daje te same wyniki co naive implementation.
    /// Test regression prevention - SIMD optimization nie może zmieniać wyników.
    /// </summary>
    [Fact]
    public void Matrix_MultiplyVector_MatchesNaiveImplementation()
    {
        var matrix = Matrix.Random(15, 20, seed: 42); // Non-standard size dla SIMD edge testing
        var vector = new double[20];
        var random = new Random(123);
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] = random.NextDouble() * 2.0 - 1.0; // [-1, 1]
        }

        var optimizedResult = matrix.MultiplyVector(vector);

        // Naive implementation for comparison
        var naiveResult = new double[matrix.Rows];
        for (int i = 0; i < matrix.Rows; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < matrix.Cols; j++)
            {
                sum += matrix[i, j] * vector[j];
            }
            naiveResult[i] = sum;
        }

        // Compare results
        optimizedResult.Length.ShouldBe(naiveResult.Length);
        for (int i = 0; i < optimizedResult.Length; i++)
        {
            optimizedResult[i].ShouldBe(naiveResult[i], 1e-12,
                $"SIMD optimization should match naive result at index {i}");
        }
    }

    /// <summary>
    /// Sprawdza error handling w matrix-vector multiplication.
    /// </summary>
    [Fact]
    public void Matrix_MultiplyVector_ValidatesVectorSize()
    {
        var matrix = new Matrix(2, 3);
        var wrongSizeVector = new double[] { 1.0, 2.0 }; // Needs 3 elements

        Should.Throw<ArgumentException>(() => matrix.MultiplyVector(wrongSizeVector))
            .Message.ShouldContain("doesn't match matrix columns");

        Should.Throw<ArgumentNullException>(() => matrix.MultiplyVector(null!));
    }

    #endregion

    #region Matrix-Matrix Multiplication Tests

    /// <summary>
    /// Testuje matrix-matrix multiplication z known mathematical example.
    /// </summary>
    [Fact]
    public void Matrix_Multiply_ComputesCorrectResult()
    {
        // A = [1 2]   B = [5 6]   Expected: A×B = [19 22]
        //     [3 4]       [7 8]                   [43 50]
        var matrixA = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var matrixB = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });

        var result = matrixA.Multiply(matrixB);

        result.Rows.ShouldBe(2);
        result.Cols.ShouldBe(2);
        result[0, 0].ShouldBe(19.0, Tolerance); // 1*5 + 2*7 = 19
        result[0, 1].ShouldBe(22.0, Tolerance); // 1*6 + 2*8 = 22
        result[1, 0].ShouldBe(43.0, Tolerance); // 3*5 + 4*7 = 43
        result[1, 1].ShouldBe(50.0, Tolerance); // 3*6 + 4*8 = 50
    }

    /// <summary>
    /// Testuje matrix multiplication dimension validation.
    /// </summary>
    [Fact]
    public void Matrix_Multiply_ValidatesDimensions()
    {
        var matrixA = new Matrix(2, 3);
        var matrixB = new Matrix(4, 2); // Cannot multiply: A.cols != B.rows

        Should.Throw<ArgumentException>(() => matrixA.Multiply(matrixB))
            .Message.ShouldContain("Cannot multiply");

        Should.Throw<ArgumentNullException>(() => matrixA.Multiply(null!));
    }

    /// <summary>
    /// Weryfikuje mathematical property: A × I = A (multiplication by identity).
    /// </summary>
    [Fact]
    public void Matrix_Multiply_IdentityProperty()
    {
        var matrix = Matrix.Random(3, 3, seed: 42);
        var identity = Matrix.Identity(3);

        var result = matrix.Multiply(identity);

        // Result should equal original matrix
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                result[i, j].ShouldBe(matrix[i, j], Tolerance,
                    $"A × I should equal A at position [{i},{j}]");
            }
        }
    }

    #endregion

    #region Transpose Tests

    /// <summary>
    /// Testuje transpose operation z known example.
    /// </summary>
    [Fact]
    public void Matrix_Transpose_ComputesCorrectResult()
    {
        var matrix = new Matrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        var transposed = matrix.Transpose();

        transposed.Rows.ShouldBe(3);
        transposed.Cols.ShouldBe(2);
        transposed[0, 0].ShouldBe(1.0, Tolerance);
        transposed[0, 1].ShouldBe(4.0, Tolerance);
        transposed[1, 0].ShouldBe(2.0, Tolerance);
        transposed[1, 1].ShouldBe(5.0, Tolerance);
        transposed[2, 0].ShouldBe(3.0, Tolerance);
        transposed[2, 1].ShouldBe(6.0, Tolerance);
    }

    /// <summary>
    /// Weryfikuje mathematical property: (A^T)^T = A.
    /// </summary>
    [Fact]
    public void Matrix_Transpose_DoubleTransposeProperty()
    {
        var matrix = Matrix.Random(4, 3, seed: 42);
        var doubleTransposed = matrix.Transpose().Transpose();

        // Should equal original matrix
        doubleTransposed.Rows.ShouldBe(matrix.Rows);
        doubleTransposed.Cols.ShouldBe(matrix.Cols);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Cols; j++)
            {
                doubleTransposed[i, j].ShouldBe(matrix[i, j], Tolerance,
                    $"(A^T)^T should equal A at position [{i},{j}]");
            }
        }
    }

    #endregion

    #region Element-wise Operations Tests

    /// <summary>
    /// Testuje element-wise addition z SIMD optimization.
    /// </summary>
    [Fact]
    public void Matrix_Add_ComputesCorrectResult()
    {
        var matrixA = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var matrixB = new Matrix(new double[,] { { 5, 6 }, { 7, 8 } });

        var result = matrixA.Add(matrixB);

        result[0, 0].ShouldBe(6.0, Tolerance); // 1 + 5
        result[0, 1].ShouldBe(8.0, Tolerance); // 2 + 6
        result[1, 0].ShouldBe(10.0, Tolerance); // 3 + 7
        result[1, 1].ShouldBe(12.0, Tolerance); // 4 + 8
    }

    /// <summary>
    /// Testuje element-wise subtraction.
    /// </summary>
    [Fact]
    public void Matrix_Subtract_ComputesCorrectResult()
    {
        var matrixA = new Matrix(new double[,] { { 10, 8 }, { 6, 4 } });
        var matrixB = new Matrix(new double[,] { { 3, 2 }, { 1, 1 } });

        var result = matrixA.Subtract(matrixB);

        result[0, 0].ShouldBe(7.0, Tolerance); // 10 - 3
        result[0, 1].ShouldBe(6.0, Tolerance); // 8 - 2
        result[1, 0].ShouldBe(5.0, Tolerance); // 6 - 1
        result[1, 1].ShouldBe(3.0, Tolerance); // 4 - 1
    }

    /// <summary>
    /// Testuje scalar multiplication.
    /// </summary>
    [Fact]
    public void Matrix_MultiplyScalar_ComputesCorrectResult()
    {
        var matrix = new Matrix(new double[,] { { 1, 2 }, { 3, 4 } });
        var scalar = 2.5;

        var result = matrix.MultiplyScalar(scalar);

        result[0, 0].ShouldBe(2.5, Tolerance); // 1 * 2.5
        result[0, 1].ShouldBe(5.0, Tolerance); // 2 * 2.5
        result[1, 0].ShouldBe(7.5, Tolerance); // 3 * 2.5
        result[1, 1].ShouldBe(10.0, Tolerance); // 4 * 2.5
    }

    /// <summary>
    /// Sprawdza dimension validation w element-wise operations.
    /// </summary>
    [Fact]
    public void Matrix_ElementWiseOps_ValidateDimensions()
    {
        var matrixA = new Matrix(2, 3);
        var matrixB = new Matrix(3, 2); // Wrong dimensions

        Should.Throw<ArgumentException>(() => matrixA.Add(matrixB));
        Should.Throw<ArgumentException>(() => matrixA.Subtract(matrixB));
    }

    /// <summary>
    /// Weryfikuje, że SIMD optimization w element-wise ops daje correct results.
    /// </summary>
    [Fact]
    public void Matrix_ElementWiseOps_SIMDOptimizationCorrectness()
    {
        // Use size that tests SIMD boundary conditions
        var matrixA = Matrix.Random(7, 13, seed: 42); // Prime numbers dla edge testing
        var matrixB = Matrix.Random(7, 13, seed: 24);

        var addResult = matrixA.Add(matrixB);
        var subResult = matrixA.Subtract(matrixB);

        // Verify against naive implementation
        for (int i = 0; i < matrixA.Rows; i++)
        {
            for (int j = 0; j < matrixA.Cols; j++)
            {
                var expectedAdd = matrixA[i, j] + matrixB[i, j];
                var expectedSub = matrixA[i, j] - matrixB[i, j];

                addResult[i, j].ShouldBe(expectedAdd, 1e-15,
                    $"SIMD add should match naive at [{i},{j}]");
                subResult[i, j].ShouldBe(expectedSub, 1e-15,
                    $"SIMD subtract should match naive at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Static Factory Methods Tests

    /// <summary>
    /// Testuje Identity matrix creation.
    /// </summary>
    [Fact]
    public void Matrix_Identity_CreatesCorrectMatrix()
    {
        var identity = Matrix.Identity(3);

        identity.Rows.ShouldBe(3);
        identity.Cols.ShouldBe(3);
        identity.IsSquare.ShouldBeTrue();

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (i == j)
                {
                    identity[i, j].ShouldBe(1.0, Tolerance, $"Diagonal element [{i},{j}] should be 1");
                }
                else
                {
                    identity[i, j].ShouldBe(0.0, Tolerance, $"Off-diagonal element [{i},{j}] should be 0");
                }
            }
        }
    }

    /// <summary>
    /// Testuje Fill method.
    /// </summary>
    [Fact]
    public void Matrix_Fill_CreatesCorrectMatrix()
    {
        var filled = Matrix.Fill(2, 3, 7.5);

        filled.Rows.ShouldBe(2);
        filled.Cols.ShouldBe(3);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                filled[i, j].ShouldBe(7.5, Tolerance, $"Element [{i},{j}] should be filled value");
            }
        }
    }

    /// <summary>
    /// Testuje Random matrix generation z reproducibility.
    /// </summary>
    [Fact]
    public void Matrix_Random_IsReproducibleWithSeed()
    {
        var matrix1 = Matrix.Random(3, 4, seed: 42);
        var matrix2 = Matrix.Random(3, 4, seed: 42);

        // Same seed should produce identical matrices
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                matrix1[i, j].ShouldBe(matrix2[i, j], Tolerance,
                    $"Same seed should produce identical values at [{i},{j}]");
            }
        }
    }

    /// <summary>
    /// Weryfikuje, że Random matrix ma reasonable statistical properties.
    /// </summary>
    [Fact]
    public void Matrix_Random_HasReasonableStatisticalProperties()
    {
        var matrix = Matrix.Random(100, 50, seed: 42);
        var values = matrix.ToArray1D();

        // Test mean ≈ 0 (Gaussian distribution)
        var mean = values.Average();
        Math.Abs(mean).ShouldBeLessThan(0.1, "Mean should be close to 0 for Gaussian distribution");

        // Test that values are varied (not all the same)
        var distinctValues = (double)values.Distinct().Count();
        distinctValues.ShouldBeGreaterThan(values.Length * 0.95, "Values should be highly varied");

        // Test scaling (Xavier initialization: scale = sqrt(1/cols))
        var expectedScale = Math.Sqrt(1.0 / 50);
        var standardDeviation = Math.Sqrt(values.Select(x => x * x).Average()); // RMS for zero-mean
        standardDeviation.ShouldBe(expectedScale, 0.1, "Standard deviation should match Xavier scaling");
    }

    #endregion

    #region Utility Methods Tests

    /// <summary>
    /// Testuje conversion methods (ToArray2D, ToArray1D).
    /// </summary>
    [Fact]
    public void Matrix_ConversionMethods_ProduceCorrectResults()
    {
        var matrix = new Matrix(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });

        var array2D = matrix.ToArray2D();
        var array1D = matrix.ToArray1D();

        // Test 2D conversion
        array2D.GetLength(0).ShouldBe(2);
        array2D.GetLength(1).ShouldBe(3);
        array2D[0, 0].ShouldBe(1.0, Tolerance);
        array2D[1, 2].ShouldBe(6.0, Tolerance);

        // Test 1D conversion (row-major)
        array1D.Length.ShouldBe(6);
        array1D[0].ShouldBe(1.0, Tolerance); // [0,0]
        array1D[2].ShouldBe(3.0, Tolerance); // [0,2]
        array1D[3].ShouldBe(4.0, Tolerance); // [1,0]
        array1D[5].ShouldBe(6.0, Tolerance); // [1,2]

        // Test defensive copying
        array2D[0, 0] = 999.0;
        matrix[0, 0].ShouldBe(1.0, Tolerance, "Conversion should be defensive copy");
    }

    /// <summary>
    /// Testuje GetRow i GetColumn methods.
    /// </summary>
    [Fact]
    public void Matrix_GetRowColumn_ExtractCorrectData()
    {
        var matrix = new Matrix(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        var row1 = matrix.GetRow(1);
        var col2 = matrix.GetColumn(2);

        row1.ShouldBe(new double[] { 4, 5, 6 }, "Row 1 should be [4, 5, 6]");
        col2.ShouldBe(new double[] { 3, 6, 9 }, "Column 2 should be [3, 6, 9]");

        // Test bounds checking
        Should.Throw<IndexOutOfRangeException>(() => matrix.GetRow(3));
        Should.Throw<IndexOutOfRangeException>(() => matrix.GetColumn(3));
        Should.Throw<IndexOutOfRangeException>(() => matrix.GetRow(-1));
        Should.Throw<IndexOutOfRangeException>(() => matrix.GetColumn(-1));
    }

    #endregion

    #region Indexer and Bounds Tests

    /// <summary>
    /// Testuje matrix indexer bounds checking.
    /// </summary>
    [Fact]
    public void Matrix_Indexer_ValidatesBounds()
    {
        var matrix = new Matrix(2, 3);

        // Valid access should work
        Should.NotThrow(() => matrix[0, 0] = 1.0);
        Should.NotThrow(() => matrix[1, 2] = 2.0);

        // Invalid access should throw
        Should.Throw<IndexOutOfRangeException>(() => matrix[2, 0]);
        Should.Throw<IndexOutOfRangeException>(() => matrix[0, 3]);
        Should.Throw<IndexOutOfRangeException>(() => matrix[-1, 0]);
        Should.Throw<IndexOutOfRangeException>(() => matrix[0, -1]);
    }

    #endregion

    #region Performance and Edge Cases

    /// <summary>
    /// Testuje behavior z single-element matrix.
    /// </summary>
    [Fact]
    public void Matrix_SingleElement_HandledCorrectly()
    {
        var matrix = new Matrix(1, 1);
        matrix[0, 0] = 42.0;

        matrix.IsSquare.ShouldBeTrue();
        matrix.IsVector.ShouldBeTrue();
        matrix.Size.ShouldBe(1);

        var vector = new double[] { 2.0 };
        var result = matrix.MultiplyVector(vector);
        result[0].ShouldBe(84.0, Tolerance); // 42 * 2
    }

    /// <summary>
    /// Testuje operations które powinny preserve zero values.
    /// </summary>
    [Fact]
    public void Matrix_ZeroOperations_PreserveZeros()
    {
        var zeros = new Matrix(3, 3);
        var identity = Matrix.Identity(3);

        // Zero matrix operations
        var zeroResult = zeros.MultiplyVector(new double[] { 1, 2, 3 });
        zeroResult.ShouldAllBe(x => Math.Abs(x) < Tolerance, "Zero matrix × vector should be zero");

        var zeroMatrix = zeros.Multiply(identity);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                zeroMatrix[i, j].ShouldBe(0.0, Tolerance, "Zero matrix × identity should be zero");
            }
        }
    }

    /// <summary>
    /// Weryfikuje ToString methods dla debugging.
    /// </summary>
    [Fact]
    public void Matrix_ToString_ProvidesUsefulInformation()
    {
        var matrix = new Matrix(3, 4);
        var description = matrix.ToString();

        description.ShouldContain("3");
        description.ShouldContain("4");
        description.ShouldContain("Matrix");

        var detailedString = matrix.ToDetailedString();
        detailedString.ShouldContain("3×4");
    }

    /// <summary>
    /// Performance smoke test - operations shouldn't be unreasonably slow.
    /// </summary>
    [Fact]
    public void Matrix_Operations_PerformanceSmoke()
    {
        var largeMatrix = Matrix.Random(500, 300, seed: 42);
        var vector = new double[300];
        Array.Fill(vector, 1.0);

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // This should complete in reasonable time (< 100ms on modern hardware)
        var result = largeMatrix.MultiplyVector(vector);
        
        stopwatch.Stop();
        
        result.Length.ShouldBe(500);
        stopwatch.ElapsedMilliseconds.ShouldBeLessThan(100, 
            "Large matrix-vector multiplication should be reasonably fast");
    }

    #endregion

    #region Integration with Neural Network Operations

    /// <summary>
    /// Testuje typical neural network operation sequence.
    /// Forward pass: activation = sigmoid(W × input + bias)
    /// </summary>
    [Fact]
    public void Matrix_NeuralNetworkIntegration_TypicalForwardPass()
    {
        // Simulate layer: 3 inputs → 2 neurons
        var weights = Matrix.Random(2, 3, seed: 42);
        var bias = new double[] { 0.1, 0.2 };
        var input = new double[] { 1.0, 0.5, -0.3 };

        // Forward pass computation
        var netInputs = weights.MultiplyVector(input);
        
        // Add bias (would normally be done in Layer class)
        for (int i = 0; i < netInputs.Length; i++)
        {
            netInputs[i] += bias[i];
        }

        // Apply activation (would normally use IActivationFunction)
        var activations = netInputs.Select(x => 1.0 / (1.0 + Math.Exp(-x))).ToArray(); // Sigmoid

        // Verify reasonable results
        activations.Length.ShouldBe(2);
        activations.ShouldAllBe(x => x > 0 && x < 1, "Sigmoid outputs should be in (0,1)");
        netInputs.ShouldAllBe(x => double.IsFinite(x), "Net inputs should be finite");
    }

    #endregion
}
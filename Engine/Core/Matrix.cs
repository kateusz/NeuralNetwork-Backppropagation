using System.Numerics;

namespace Engine.Core;

/// <summary>
/// Macierz zoptymalizowana dla operacji w sieciach neuronowych.
/// Wykorzystuje System.Numerics.Vector&lt;double&gt; dla SIMD optimization.
/// 
/// Przechowywanie: row-major order (standardowe dla .NET)
/// Indeksowanie: matrix[row, col] 
/// 
/// Kluczowe operacje dla ML:
/// - Matrix × Vector (forward propagation)  
/// - Matrix × Matrix (weight updates, transformations)
/// - Transpose (backward propagation)
/// - Element-wise operations (activation functions, gradients)
/// </summary>
public class Matrix
{
    #region Private Fields
    
    /// <summary>
    /// Dane macierzy w formacie row-major: [row0_col0, row0_col1, ..., row0_colN, row1_col0, ...]
    /// Wykorzystujemy double[] dla compatibility z Vector&lt;double&gt; SIMD operations.
    /// </summary>
    private readonly double[] _data;
    
    /// <summary>
    /// Liczba wierszy macierzy.
    /// </summary>
    private readonly int _rows;
    
    /// <summary>
    /// Liczba kolumn macierzy.
    /// </summary>
    private readonly int _cols;
    
    #endregion
    
    #region Public Properties
    
    /// <summary>
    /// Liczba wierszy w macierzy.
    /// </summary>
    public int Rows => _rows;
    
    /// <summary>
    /// Liczba kolumn w macierzy.
    /// </summary>
    public int Cols => _cols;
    
    /// <summary>
    /// Całkowita liczba elementów w macierzy.
    /// </summary>
    public int Size => _rows * _cols;
    
    /// <summary>
    /// Czy macierz jest kwadratowa (rows == cols).
    /// </summary>
    public bool IsSquare => _rows == _cols;
    
    /// <summary>
    /// Czy macierz jest wektorem (single row lub single column).
    /// </summary>
    public bool IsVector => _rows == 1 || _cols == 1;
    
    /// <summary>
    /// Indekser dla dostępu do elementów macierzy.
    /// Używa row-major indexing: index = row * _cols + col
    /// </summary>
    /// <param name="row">Indeks wiersza (0-based)</param>
    /// <param name="col">Indeks kolumny (0-based)</param>
    /// <returns>Wartość elementu w pozycji [row, col]</returns>
    /// <exception cref="IndexOutOfRangeException">Gdy indeksy są poza zakresem</exception>
    public double this[int row, int col]
    {
        get
        {
            ValidateIndices(row, col);
            return _data[row * _cols + col];
        }
        set
        {
            ValidateIndices(row, col);
            _data[row * _cols + col] = value;
        }
    }
    
    #endregion
    
    #region Constructors
    
    /// <summary>
    /// Tworzy nową macierz o określonych wymiarach wypełnioną zerami.
    /// </summary>
    /// <param name="rows">Liczba wierszy</param>
    /// <param name="cols">Liczba kolumn</param>
    /// <exception cref="ArgumentException">Gdy wymiary są nieprawidłowe</exception>
    public Matrix(int rows, int cols)
    {
        if (rows <= 0)
            throw new ArgumentException("Number of rows must be positive", nameof(rows));
        if (cols <= 0)
            throw new ArgumentException("Number of columns must be positive", nameof(cols));
        
        _rows = rows;
        _cols = cols;
        _data = new double[rows * cols];
    }
    
    /// <summary>
    /// Tworzy macierz z dwuwymiarowej tablicy.
    /// </summary>
    /// <param name="data">Dane macierzy w formacie [row, col]</param>
    /// <exception cref="ArgumentNullException">Gdy data jest null</exception>
    /// <exception cref="ArgumentException">Gdy data jest pusta</exception>
    public Matrix(double[,] data)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        
        _rows = data.GetLength(0);
        _cols = data.GetLength(1);
        
        if (_rows == 0 || _cols == 0)
            throw new ArgumentException("Matrix dimensions must be positive", nameof(data));
        
        _data = new double[_rows * _cols];
        
        // Kopiuj z row-major order
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                _data[i * _cols + j] = data[i, j];
            }
        }
    }
    
    /// <summary>
    /// Tworzy macierz z jednowymiarowej tablicy w formacie row-major.
    /// </summary>
    /// <param name="data">Dane w formacie row-major</param>
    /// <param name="rows">Liczba wierszy</param>
    /// <param name="cols">Liczba kolumn</param>
    /// <exception cref="ArgumentNullException">Gdy data jest null</exception>
    /// <exception cref="ArgumentException">Gdy rozmiary nie pasują do danych</exception>
    public Matrix(double[] data, int rows, int cols)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (rows <= 0)
            throw new ArgumentException("Number of rows must be positive", nameof(rows));
        if (cols <= 0)
            throw new ArgumentException("Number of columns must be positive", nameof(cols));
        if (data.Length != rows * cols)
            throw new ArgumentException(
                $"Data length {data.Length} doesn't match matrix size {rows}×{cols}={rows * cols}",
                nameof(data));
        
        _rows = rows;
        _cols = cols;
        _data = new double[data.Length];
        Array.Copy(data, _data, data.Length);
    }
    
    #endregion
    
    #region Core Matrix Operations
    
    /// <summary>
    /// Mnoży macierz przez wektor - kluczowa operacja w neural networks.
    /// Implementuje: result = this × vector
    /// 
    /// Używa SIMD optimization przez Vector&lt;double&gt; gdzie to możliwe.
    /// Complexity: O(rows × cols) z SIMD acceleration.
    /// </summary>
    /// <param name="vector">Wektor do pomnożenia [cols×1]</param>
    /// <returns>Wynikowy wektor [rows×1]</returns>
    /// <exception cref="ArgumentNullException">Gdy vector jest null</exception>
    /// <exception cref="ArgumentException">Gdy wymiary nie pasują</exception>
    public double[] MultiplyVector(double[] vector)
    {
        if (vector == null)
            throw new ArgumentNullException(nameof(vector));
        if (vector.Length != _cols)
            throw new ArgumentException(
                $"Vector length {vector.Length} doesn't match matrix columns {_cols}",
                nameof(vector));
        
        var result = new double[_rows];
        var vectorLength = Vector<double>.Count;
        
        for (int row = 0; row < _rows; row++)
        {
            double sum = 0.0;
            int rowStart = row * _cols;
            
            // SIMD-optimized portion - process chunks of Vector<double>.Count elements
            int vectorizedLength = (_cols / vectorLength) * vectorLength;
            
            for (int col = 0; col < vectorizedLength; col += vectorLength)
            {
                // Load matrix row segment and vector segment
                var matrixSegment = new double[vectorLength];
                var vectorSegment = new double[vectorLength];
                
                Array.Copy(_data, rowStart + col, matrixSegment, 0, vectorLength);
                Array.Copy(vector, col, vectorSegment, 0, vectorLength);
                
                // SIMD dot product
                var matrixVector = new Vector<double>(matrixSegment);
                var inputVector = new Vector<double>(vectorSegment);
                sum += Vector.Dot(matrixVector, inputVector);
            }
            
            // Handle remaining elements (non-vectorized tail)
            for (int col = vectorizedLength; col < _cols; col++)
            {
                sum += _data[rowStart + col] * vector[col];
            }
            
            result[row] = sum;
        }
        
        return result;
    }
    
    /// <summary>
    /// Mnoży dwie macierze: result = this × other
    /// 
    /// Używa SIMD optimization dla każdego row×column dot product.
    /// Complexity: O(this.Rows × this.Cols × other.Cols) z SIMD acceleration.
    /// </summary>
    /// <param name="other">Macierz do pomnożenia</param>
    /// <returns>Wynikowa macierz [this.Rows × other.Cols]</returns>
    /// <exception cref="ArgumentNullException">Gdy other jest null</exception>
    /// <exception cref="ArgumentException">Gdy wymiary nie pozwalają na mnożenie</exception>
    public Matrix Multiply(Matrix other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        if (_cols != other._rows)
            throw new ArgumentException(
                $"Cannot multiply {_rows}×{_cols} matrix by {other._rows}×{other._cols} matrix",
                nameof(other));
        
        var result = new Matrix(_rows, other._cols);
        
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < other._cols; j++)
            {
                double sum = 0.0;
                
                // Compute dot product of row i from this and column j from other
                for (int k = 0; k < _cols; k++)
                {
                    sum += this[i, k] * other[k, j];
                }
                
                result[i, j] = sum;
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Zwraca transpozycję macierzy: A^T
    /// Kluczowa operacja dla backward propagation w neural networks.
    /// 
    /// Complexity: O(rows × cols) - single pass przez wszystkie elementy.
    /// </summary>
    /// <returns>Transponowana macierz [cols × rows]</returns>
    public Matrix Transpose()
    {
        var result = new Matrix(_cols, _rows);
        
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                // A^T[j,i] = A[i,j]
                result[j, i] = this[i, j];
            }
        }
        
        return result;
    }
    
    #endregion
    
    #region Element-wise Operations
    
    /// <summary>
    /// Dodaje dwie macierze element-wise: result = this + other
    /// </summary>
    /// <param name="other">Macierz do dodania</param>
    /// <returns>Wynikowa macierz</returns>
    /// <exception cref="ArgumentNullException">Gdy other jest null</exception>
    /// <exception cref="ArgumentException">Gdy wymiary nie pasują</exception>
    public Matrix Add(Matrix other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        if (_rows != other._rows || _cols != other._cols)
            throw new ArgumentException(
                $"Cannot add {_rows}×{_cols} matrix to {other._rows}×{other._cols} matrix",
                nameof(other));
        
        var result = new Matrix(_rows, _cols);
        var vectorLength = Vector<double>.Count;
        var vectorizedLength = (_data.Length / vectorLength) * vectorLength;
        
        // SIMD-optimized addition
        for (int i = 0; i < vectorizedLength; i += vectorLength)
        {
            var thisSegment = new double[vectorLength];
            var otherSegment = new double[vectorLength];
            
            Array.Copy(_data, i, thisSegment, 0, vectorLength);
            Array.Copy(other._data, i, otherSegment, 0, vectorLength);
            
            var thisVector = new Vector<double>(thisSegment);
            var otherVector = new Vector<double>(otherSegment);
            var resultVector = thisVector + otherVector;
            
            resultVector.CopyTo(result._data, i);
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < _data.Length; i++)
        {
            result._data[i] = _data[i] + other._data[i];
        }
        
        return result;
    }
    
    /// <summary>
    /// Odejmuje macierze element-wise: result = this - other
    /// </summary>
    /// <param name="other">Macierz do odjęcia</param>
    /// <returns>Wynikowa macierz</returns>
    /// <exception cref="ArgumentNullException">Gdy other jest null</exception>
    /// <exception cref="ArgumentException">Gdy wymiary nie pasują</exception>
    public Matrix Subtract(Matrix other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        if (_rows != other._rows || _cols != other._cols)
            throw new ArgumentException(
                $"Cannot subtract {other._rows}×{other._cols} matrix from {_rows}×{_cols} matrix",
                nameof(other));
        
        var result = new Matrix(_rows, _cols);
        var vectorLength = Vector<double>.Count;
        var vectorizedLength = (_data.Length / vectorLength) * vectorLength;
        
        // SIMD-optimized subtraction
        for (int i = 0; i < vectorizedLength; i += vectorLength)
        {
            var thisSegment = new double[vectorLength];
            var otherSegment = new double[vectorLength];
            
            Array.Copy(_data, i, thisSegment, 0, vectorLength);
            Array.Copy(other._data, i, otherSegment, 0, vectorLength);
            
            var thisVector = new Vector<double>(thisSegment);
            var otherVector = new Vector<double>(otherSegment);
            var resultVector = thisVector - otherVector;
            
            resultVector.CopyTo(result._data, i);
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < _data.Length; i++)
        {
            result._data[i] = _data[i] - other._data[i];
        }
        
        return result;
    }
    
    /// <summary>
    /// Mnoży macierz przez skalar: result = this × scalar
    /// </summary>
    /// <param name="scalar">Skalar do pomnożenia</param>
    /// <returns>Wynikowa macierz</returns>
    public Matrix MultiplyScalar(double scalar)
    {
        var result = new Matrix(_rows, _cols);
        var vectorLength = Vector<double>.Count;
        var vectorizedLength = (_data.Length / vectorLength) * vectorLength;
        var scalarVector = new Vector<double>(scalar);
        
        // SIMD-optimized scalar multiplication
        for (int i = 0; i < vectorizedLength; i += vectorLength)
        {
            var dataSegment = new double[vectorLength];
            Array.Copy(_data, i, dataSegment, 0, vectorLength);
            
            var dataVector = new Vector<double>(dataSegment);
            var resultVector = dataVector * scalarVector;
            
            resultVector.CopyTo(result._data, i);
        }
        
        // Handle remaining elements
        for (int i = vectorizedLength; i < _data.Length; i++)
        {
            result._data[i] = _data[i] * scalar;
        }
        
        return result;
    }
    
    #endregion
    
    #region Static Factory Methods
    
    /// <summary>
    /// Tworzy macierz jednostkową (identity matrix) o określonym rozmiarze.
    /// </summary>
    /// <param name="size">Rozmiar macierzy (size × size)</param>
    /// <returns>Macierz jednostkowa</returns>
    /// <exception cref="ArgumentException">Gdy size jest nieprawidłowy</exception>
    public static Matrix Identity(int size)
    {
        if (size <= 0)
            throw new ArgumentException("Matrix size must be positive", nameof(size));
        
        var result = new Matrix(size, size);
        for (int i = 0; i < size; i++)
        {
            result[i, i] = 1.0;
        }
        return result;
    }
    
    /// <summary>
    /// Tworzy macierz wypełnioną określoną wartością.
    /// </summary>
    /// <param name="rows">Liczba wierszy</param>
    /// <param name="cols">Liczba kolumn</param>
    /// <param name="value">Wartość wypełniająca</param>
    /// <returns>Wypełniona macierz</returns>
    /// <exception cref="ArgumentException">Gdy wymiary są nieprawidłowe</exception>
    public static Matrix Fill(int rows, int cols, double value)
    {
        var result = new Matrix(rows, cols);
        if (value != 0.0) // Optimization: zeros are already filled by constructor
        {
            Array.Fill(result._data, value);
        }
        return result;
    }
    
    /// <summary>
    /// Tworzy macierz z losowymi wartościami z rozkładu normalnego N(0, 1).
    /// Używa Xavier/Glorot initialization scaling.
    /// </summary>
    /// <param name="rows">Liczba wierszy</param>
    /// <param name="cols">Liczba kolumn</param>
    /// <param name="seed">Seed dla reproducibility (optional)</param>
    /// <returns>Macierz z losowymi wartościami</returns>
    /// <exception cref="ArgumentException">Gdy wymiary są nieprawidłowe</exception>
    public static Matrix Random(int rows, int cols, int? seed = null)
    {
        if (rows <= 0)
            throw new ArgumentException("Number of rows must be positive", nameof(rows));
        if (cols <= 0)
            throw new ArgumentException("Number of columns must be positive", nameof(cols));
        
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var result = new Matrix(rows, cols);
        
        // Xavier/Glorot initialization: scale = sqrt(1 / fan_in)
        double scale = Math.Sqrt(1.0 / cols);
        
        for (int i = 0; i < result._data.Length; i++)
        {
            // Box-Muller transformation for Gaussian distribution
            result._data[i] = GenerateGaussianRandom(random) * scale;
        }
        
        return result;
    }
    
    #endregion
    
    #region Utility Methods
    
    /// <summary>
    /// Kopiuje macierz do dwuwymiarowej tablicy.
    /// </summary>
    /// <returns>Kopia danych w formacie [row, col]</returns>
    public double[,] ToArray2D()
    {
        var result = new double[_rows, _cols];
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < _cols; j++)
            {
                result[i, j] = this[i, j];
            }
        }
        return result;
    }
    
    /// <summary>
    /// Kopiuje macierz do jednowymiarowej tablicy w formacie row-major.
    /// </summary>
    /// <returns>Kopia danych w formacie row-major</returns>
    public double[] ToArray1D()
    {
        var result = new double[_data.Length];
        Array.Copy(_data, result, _data.Length);
        return result;
    }
    
    /// <summary>
    /// Zwraca określony wiersz jako wektor.
    /// </summary>
    /// <param name="row">Indeks wiersza</param>
    /// <returns>Kopia wiersza jako array</returns>
    /// <exception cref="IndexOutOfRangeException">Gdy row jest poza zakresem</exception>
    public double[] GetRow(int row)
    {
        if (row < 0 || row >= _rows)
            throw new IndexOutOfRangeException($"Row index {row} is out of range [0, {_rows})");
        
        var result = new double[_cols];
        Array.Copy(_data, row * _cols, result, 0, _cols);
        return result;
    }
    
    /// <summary>
    /// Zwraca określoną kolumnę jako wektor.
    /// </summary>
    /// <param name="col">Indeks kolumny</param>
    /// <returns>Kopia kolumny jako array</returns>
    /// <exception cref="IndexOutOfRangeException">Gdy col jest poza zakresem</exception>
    public double[] GetColumn(int col)
    {
        if (col < 0 || col >= _cols)
            throw new IndexOutOfRangeException($"Column index {col} is out of range [0, {_cols})");
        
        var result = new double[_rows];
        for (int i = 0; i < _rows; i++)
        {
            result[i] = this[i, col];
        }
        return result;
    }
    
    #endregion
    
    #region Private Helpers
    
    /// <summary>
    /// Waliduje indeksy dostępu do macierzy.
    /// </summary>
    /// <param name="row">Indeks wiersza</param>
    /// <param name="col">Indeks kolumny</param>
    /// <exception cref="IndexOutOfRangeException">Gdy indeksy są nieprawidłowe</exception>
    private void ValidateIndices(int row, int col)
    {
        if (row < 0 || row >= _rows)
            throw new IndexOutOfRangeException($"Row index {row} is out of range [0, {_rows})");
        if (col < 0 || col >= _cols)
            throw new IndexOutOfRangeException($"Column index {col} is out of range [0, {_cols})");
    }
    
    /// <summary>
    /// Generuje liczbę losową z rozkładu normalnego N(0,1) używając Box-Muller transformation.
    /// </summary>
    /// <param name="random">Generator liczb losowych</param>
    /// <returns>Losowa liczba z rozkładu normalnego</returns>
    private static double GenerateGaussianRandom(Random random)
    {
        // Box-Muller transformation
        double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] 
        double u2 = 1.0 - random.NextDouble(); // Uniform(0,1]
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
    
    #endregion
    
    #region Debug and Display
    
    /// <summary>
    /// Zwraca string reprezentację macierzy dla debugowania.
    /// </summary>
    /// <returns>Formatted string representation</returns>
    public override string ToString()
    {
        return $"Matrix({_rows}×{_cols})";
    }
    
    /// <summary>
    /// Zwraca szczegółową reprezentację macierzy z wartościami (dla małych macierzy).
    /// </summary>
    /// <param name="maxElements">Maksymalna liczba elementów do wyświetlenia</param>
    /// <returns>Detailed string representation</returns>
    public string ToDetailedString(int maxElements = 100)
    {
        if (Size > maxElements)
        {
            return $"Matrix({_rows}×{_cols}) [too large to display, {Size} elements]";
        }
        
        var lines = new List<string>();
        lines.Add($"Matrix({_rows}×{_cols}):");
        
        for (int i = 0; i < _rows; i++)
        {
            var rowValues = new List<string>();
            for (int j = 0; j < _cols; j++)
            {
                rowValues.Add($"{this[i, j]:F4}");
            }
            lines.Add($"  [{string.Join(", ", rowValues)}]");
        }
        
        return string.Join(Environment.NewLine, lines);
    }
    
    #endregion
}
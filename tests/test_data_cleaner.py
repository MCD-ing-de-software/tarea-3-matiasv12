import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 500],  # 200 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos.
        
        Este test demuestra cómo usar pandas.testing.assert_frame_equal() para comparar
        DataFrames completos, lo cual es útil porque maneja correctamente los índices,
        tipos de datos y valores NaN de Pandas.
        """
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        # DataFrame esperado después de trim
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        
        # Usar pandas.testing.assert_frame_equal() para comparar DataFrames completos
        # Esto maneja correctamente índices, tipos y estructura de Pandas
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series.
        
        Este test demuestra cómo usar pandas.testing.assert_series_equal() para comparar
        Series completas, útil cuando queremos verificar que una columna completa tiene
        los valores esperados manteniendo los índices correctos.
        """
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Verificar que la columna 'name' ya no tiene valores faltantes
        # Los índices después de drop_invalid_rows son [0, 2] (se eliminó la fila 1)
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        
        # Usar pandas.testing.assert_series_equal() para comparar Series completas
        # Esto verifica valores, índices y tipos correctamente
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que el método drop_invalid_rows elimina correctamente las filas
        que contienen valores faltantes (NaN o None) en las columnas especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con valores faltantes usando make_sample_df()
        - Llamar a drop_invalid_rows con las columnas "name" y "age"
        - Verificar que el DataFrame resultante no tiene valores faltantes en esas columnas
        - Verificar que el DataFrame resultante tiene menos filas que el original
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name", "age"])
        
        # Verificar que no hay valores faltantes en las columnas especificadas
        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)
        
        # Verificar que el DataFrame resultante tiene menos filas que el original
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que el método drop_invalid_rows lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a drop_invalid_rows con una columna que no existe
        - Verificar que se lanza un KeyError
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica que el método trim_strings elimina correctamente los espacios
        en blanco al inicio y final de los valores en las columnas especificadas, sin modificar
        el DataFrame original ni las columnas no especificadas.
        
        Escenario esperado:
        - Crear un DataFrame con espacios en blanco usando make_sample_df()
        - Llamar a trim_strings con la columna "name"
        - Verificar que el DataFrame original no fue modificado
        - Verificar que en el DataFrame resultante los valores de "name" no tienen espacios
        - Verificar que las columnas no especificadas permanecen sin cambios
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        # Guardar el valor original para verificar que no se modifica
        original_first_name = df["name"].iloc[0]
        
        result = cleaner.trim_strings(df, ["name"])
        
        # Verificar que el DataFrame original no fue modificado
        self.assertEqual(df["name"].iloc[0], original_first_name)
        
        # Verificar que los espacios fueron eliminados en el resultado
        self.assertEqual(result["name"].iloc[0], "Alice")
        self.assertEqual(result["name"].iloc[3], "Carol")
        
        # Verificar que la columna "city" no cambió
        pdt.assert_series_equal(result["city"], df["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que el método trim_strings lanza un TypeError cuando
        se llama con una columna que no es de tipo string.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a trim_strings con una columna numérica
        - Verificar que se lanza un TypeError
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica que el método remove_outliers_iqr elimina correctamente los
        valores extremos (outliers) de una columna numérica usando el método del rango
        intercuartílico (IQR).
        
        Escenario esperado:
        - Crear un DataFrame con valores extremos usando make_sample_df()
        - Llamar a remove_outliers_iqr con la columna "age" y factor=1.5
        - Verificar que el valor extremo (200) fue eliminado
        - Verificar que valores normales permanecen
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)
        
        # Verificar que el outlier (500) fue eliminado
        self.assertNotIn(500, result["age"].values)
        
        # Verificar que al menos uno de los valores normales permanece
        self.assertIn(25, result["age"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un KeyError cuando
        se llama con una columna que no existe en el DataFrame.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna que no existe
        - Verificar que se lanza un KeyError
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary", factor=1.5)

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que el método remove_outliers_iqr lanza un TypeError cuando
        se llama con una columna que no es de tipo numérico.
        
        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a remove_outliers_iqr con una columna de texto
        - Verificar que se lanza un TypeError
        """
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city", factor=1.5)


if __name__ == "__main__":
    unittest.main()
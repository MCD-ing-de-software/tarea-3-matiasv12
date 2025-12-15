import pandas as pd
import numpy as np


class DataCleaner:
    """Clase para limpieza y preprocesamiento de DataFrames de Pandas."""

    def remove_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina espacios en blanco al inicio y final de strings en todas las columnas object.
        
        Args:
            df: DataFrame a procesar
            
        Returns:
            DataFrame con strings sin espacios en blanco
        """
        result = df.copy()
        
        # Aplicar strip solo a columnas de tipo object/string
        for col in result.select_dtypes(include=['object']).columns:
            result[col] = result[col].str.strip()
        
        return result

    def drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas con valores faltantes en cualquier columna.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame sin filas con valores faltantes
        """
        return df.dropna().copy()

    def fill_nulls_numeric(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Rellena valores faltantes con la media de la columna numérica.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna numérica
            
        Returns:
            DataFrame con valores faltantes rellenados con la media
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        result = df.copy()
        mean_val = result[column].mean()
        result[column] = result[column].fillna(mean_val)
        
        return result

    def remove_outliers_iqr(self, df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
        """Elimina outliers usando el método del rango intercuartílico (IQR).
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna numérica
            factor: Factor multiplicador del IQR (default 1.5)
            
        Returns:
            DataFrame sin outliers en la columna especificada
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        # Calcular IQR solo con valores válidos (no NaN)
        valid_values = df[column].dropna()
        
        if len(valid_values) == 0:
            return df.copy()
        
        Q1 = valid_values.quantile(0.25)
        Q3 = valid_values.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Mantener valores dentro del rango O valores NaN
        mask = ((df[column] >= lower_bound) & (df[column] <= upper_bound)) | df[column].isna()
        
        return df[mask].copy()

    def drop_invalid_rows(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Elimina filas con valores faltantes en las columnas especificadas.
        
        Args:
            df: DataFrame a limpiar
            columns: Lista de nombres de columnas a verificar
            
        Returns:
            DataFrame sin filas con valores faltantes en las columnas especificadas
            
        Raises:
            KeyError: Si alguna columna no existe en el DataFrame
        """
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
        
        return df.dropna(subset=columns).copy()

    def trim_strings(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Elimina espacios en blanco al inicio y final de strings.
        
        Args:
            df: DataFrame a procesar
            columns: Lista de columnas de tipo string a limpiar
            
        Returns:
            DataFrame con strings sin espacios en blanco
            
        Raises:
            KeyError: Si alguna columna no existe en el DataFrame
            TypeError: Si alguna columna no es de tipo string/object
        """
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not of string type")
            
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                raise TypeError(f"Column '{col}' is not of string type")
            
            result[col] = df[col].str.strip()
        
        return result

    def normalize_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Normaliza una columna numérica usando Min-Max scaling (0-1).
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna a normalizar
            
        Returns:
            DataFrame con la columna normalizada
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        result = df.copy()
        
        min_val = result[column].min()
        max_val = result[column].max()
        
        if max_val == min_val:
            result[column] = 0.0
        else:
            result[column] = (result[column] - min_val) / (max_val - min_val)
        
        return result

    def standardize_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Estandariza una columna numérica usando Z-score (media=0, std=1).
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna a estandarizar
            
        Returns:
            DataFrame con la columna estandarizada
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        result = df.copy()
        
        mean_val = result[column].mean()
        std_val = result[column].std()
        
        if std_val == 0:
            result[column] = 0.0
        else:
            result[column] = (result[column] - mean_val) / std_val
        
        return result

    def fill_missing_with_mean(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Rellena valores faltantes con la media de la columna.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna
            
        Returns:
            DataFrame con valores faltantes rellenados
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        result = df.copy()
        mean_val = result[column].mean()
        result[column] = result[column].fillna(mean_val)
        
        return result

    def fill_missing_with_median(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Rellena valores faltantes con la mediana de la columna.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna
            
        Returns:
            DataFrame con valores faltantes rellenados
            
        Raises:
            KeyError: Si la columna no existe
            TypeError: Si la columna no es numérica
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not numeric")
        
        result = df.copy()
        median_val = result[column].median()
        result[column] = result[column].fillna(median_val)
        
        return result

    def fill_missing_with_mode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Rellena valores faltantes con la moda de la columna.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna
            
        Returns:
            DataFrame con valores faltantes rellenados
            
        Raises:
            KeyError: Si la columna no existe
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        result = df.copy()
        mode_val = result[column].mode()
        
        if len(mode_val) > 0:
            result[column] = result[column].fillna(mode_val[0])
        
        return result

    def convert_to_datetime(self, df: pd.DataFrame, column: str, format: str = None) -> pd.DataFrame:
        """Convierte una columna a tipo datetime.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna
            format: Formato de fecha (opcional)
            
        Returns:
            DataFrame con la columna convertida a datetime
            
        Raises:
            KeyError: Si la columna no existe
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        result = df.copy()
        result[column] = pd.to_datetime(result[column], format=format, errors='coerce')
        
        return result

    def encode_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Codifica una columna categórica usando Label Encoding.
        
        Args:
            df: DataFrame a procesar
            column: Nombre de la columna categórica
            
        Returns:
            DataFrame con la columna codificada
            
        Raises:
            KeyError: Si la columna no existe
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        result = df.copy()
        result[column] = pd.Categorical(result[column]).codes
        
        return result
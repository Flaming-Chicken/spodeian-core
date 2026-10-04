use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use thiserror::Error;

#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Error, Debug)]
pub enum ParquetError {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Column length mismatch: expected {expected} rows, but column '{column}' has {actual}")]
    LengthMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },
    #[error("Unsupported or missing column: {0}")]
    ColumnNotFound(String),
}

/// Generic columnar tabular dataset for financial indicators, time-series, and ML features.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TabularDataset {
    pub column_names: Vec<String>,
    pub float_columns: HashMap<String, Vec<f64>>,
    pub int_columns: HashMap<String, Vec<i64>>,
    pub string_columns: HashMap<String, Vec<String>>,
    pub num_rows: usize,
}

impl TabularDataset {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a float64 column.
    pub fn with_float_column(mut self, name: &str, data: Vec<f64>) -> Result<Self, ParquetError> {
        self.add_float_column(name, data)?;
        Ok(self)
    }

    /// Appends an int64 column.
    pub fn with_int_column(mut self, name: &str, data: Vec<i64>) -> Result<Self, ParquetError> {
        self.add_int_column(name, data)?;
        Ok(self)
    }

    /// Appends a string column.
    pub fn with_string_column(
        mut self,
        name: &str,
        data: Vec<String>,
    ) -> Result<Self, ParquetError> {
        self.add_string_column(name, data)?;
        Ok(self)
    }

    pub fn add_float_column(&mut self, name: &str, data: Vec<f64>) -> Result<(), ParquetError> {
        if self.column_names.is_empty() {
            self.num_rows = data.len();
        } else if data.len() != self.num_rows {
            return Err(ParquetError::LengthMismatch {
                column: name.to_string(),
                expected: self.num_rows,
                actual: data.len(),
            });
        }
        if !self.column_names.contains(&name.to_string()) {
            self.column_names.push(name.to_string());
        }
        self.float_columns.insert(name.to_string(), data);
        Ok(())
    }

    pub fn add_int_column(&mut self, name: &str, data: Vec<i64>) -> Result<(), ParquetError> {
        if self.column_names.is_empty() {
            self.num_rows = data.len();
        } else if data.len() != self.num_rows {
            return Err(ParquetError::LengthMismatch {
                column: name.to_string(),
                expected: self.num_rows,
                actual: data.len(),
            });
        }
        if !self.column_names.contains(&name.to_string()) {
            self.column_names.push(name.to_string());
        }
        self.int_columns.insert(name.to_string(), data);
        Ok(())
    }

    pub fn add_string_column(&mut self, name: &str, data: Vec<String>) -> Result<(), ParquetError> {
        if self.column_names.is_empty() {
            self.num_rows = data.len();
        } else if data.len() != self.num_rows {
            return Err(ParquetError::LengthMismatch {
                column: name.to_string(),
                expected: self.num_rows,
                actual: data.len(),
            });
        }
        if !self.column_names.contains(&name.to_string()) {
            self.column_names.push(name.to_string());
        }
        self.string_columns.insert(name.to_string(), data);
        Ok(())
    }

    /// Concatenates rows from another `TabularDataset` with matching column schema.
    pub fn extend_from(&mut self, other: &TabularDataset) -> Result<(), ParquetError> {
        if other.num_rows == 0 {
            return Ok(());
        }
        if self.column_names.is_empty() {
            *self = other.clone();
            return Ok(());
        }

        // Validate schema
        for name in &self.column_names {
            if !other.column_names.contains(name) {
                return Err(ParquetError::ColumnNotFound(name.clone()));
            }
        }

        for (col, data) in &other.float_columns {
            if let Some(target) = self.float_columns.get_mut(col) {
                target.extend_from_slice(data);
            }
        }
        for (col, data) in &other.int_columns {
            if let Some(target) = self.int_columns.get_mut(col) {
                target.extend_from_slice(data);
            }
        }
        for (col, data) in &other.string_columns {
            if let Some(target) = self.string_columns.get_mut(col) {
                target.extend_from_slice(data);
            }
        }

        self.num_rows += other.num_rows;
        Ok(())
    }

    /// Converts tabular dataset into an Apache Arrow `RecordBatch`.
    pub fn to_record_batch(&self) -> Result<RecordBatch, ParquetError> {
        let mut fields = Vec::with_capacity(self.column_names.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.column_names.len());

        for col_name in &self.column_names {
            if let Some(float_data) = self.float_columns.get(col_name) {
                fields.push(Field::new(col_name, DataType::Float64, false));
                let array = Arc::new(Float64Array::from(float_data.clone())) as ArrayRef;
                arrays.push(array);
            } else if let Some(int_data) = self.int_columns.get(col_name) {
                fields.push(Field::new(col_name, DataType::Int64, false));
                let array = Arc::new(Int64Array::from(int_data.clone())) as ArrayRef;
                arrays.push(array);
            } else if let Some(str_data) = self.string_columns.get(col_name) {
                fields.push(Field::new(col_name, DataType::Utf8, false));
                let array = Arc::new(StringArray::from_iter_values(
                    str_data.iter().map(|s| s.as_str()),
                )) as ArrayRef;
                arrays.push(array);
            } else {
                return Err(ParquetError::ColumnNotFound(col_name.clone()));
            }
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)?;
        Ok(batch)
    }

    /// Converts an Apache Arrow `RecordBatch` back into `TabularDataset`.
    pub fn from_record_batch(batch: &RecordBatch) -> Result<Self, ParquetError> {
        let num_rows = batch.num_rows();
        let mut dataset = Self {
            column_names: Vec::new(),
            float_columns: HashMap::new(),
            int_columns: HashMap::new(),
            string_columns: HashMap::new(),
            num_rows,
        };

        let schema = batch.schema();
        for (i, field) in schema.fields().iter().enumerate() {
            let col_name = field.name().clone();
            dataset.column_names.push(col_name.clone());
            let col = batch.column(i);

            match field.data_type() {
                DataType::Float64 => {
                    let float_arr =
                        col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                            ParquetError::ColumnNotFound(format!(
                                "Failed to downcast float column: {col_name}"
                            ))
                        })?;
                    let values: Vec<f64> = (0..num_rows).map(|r| float_arr.value(r)).collect();
                    dataset.float_columns.insert(col_name, values);
                }
                DataType::Int64 => {
                    let int_arr = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                        ParquetError::ColumnNotFound(format!(
                            "Failed to downcast int column: {col_name}"
                        ))
                    })?;
                    let values: Vec<i64> = (0..num_rows).map(|r| int_arr.value(r)).collect();
                    dataset.int_columns.insert(col_name, values);
                }
                DataType::Utf8 => {
                    let str_arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                        ParquetError::ColumnNotFound(format!(
                            "Failed to downcast string column: {col_name}"
                        ))
                    })?;
                    let values: Vec<String> = (0..num_rows)
                        .map(|r| str_arr.value(r).to_string())
                        .collect();
                    dataset.string_columns.insert(col_name, values);
                }
                other => {
                    return Err(ParquetError::ColumnNotFound(format!(
                        "Unsupported column data type: {other:?} for {col_name}"
                    )));
                }
            }
        }

        Ok(dataset)
    }

    /// Serializes dataset into standard Apache Parquet format.
    pub fn write_to_parquet<W: Write + Send>(&self, writer: W) -> Result<(), ParquetError> {
        let batch = self.to_record_batch()?;
        let schema = batch.schema();
        let props = WriterProperties::builder().build();

        let mut arrow_writer = ArrowWriter::try_new(writer, schema, Some(props))?;
        arrow_writer.write(&batch)?;
        arrow_writer.close()?;
        Ok(())
    }

    /// Deserializes a standard Apache Parquet byte slice into `TabularDataset`.
    pub fn read_from_parquet(bytes: &[u8]) -> Result<Self, ParquetError> {
        let chunk = bytes::Bytes::copy_from_slice(bytes);
        Self::read_from_parquet_bytes(chunk)
    }

    /// Deserializes standard Apache Parquet from `bytes::Bytes`.
    pub fn read_from_parquet_bytes(bytes: bytes::Bytes) -> Result<Self, ParquetError> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let mut arrow_reader = builder.build()?;

        if let Some(first_batch) = arrow_reader.next() {
            let batch = first_batch?;
            Self::from_record_batch(&batch)
        } else {
            Ok(Self::new())
        }
    }

    /// Deserializes standard Apache Parquet directly from a `std::fs::File`.
    pub fn read_from_parquet_file(file: std::fs::File) -> Result<Self, ParquetError> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut arrow_reader = builder.build()?;

        if let Some(first_batch) = arrow_reader.next() {
            let batch = first_batch?;
            Self::from_record_batch(&batch)
        } else {
            Ok(Self::new())
        }
    }
}

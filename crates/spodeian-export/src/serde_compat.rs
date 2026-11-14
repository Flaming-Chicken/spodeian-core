//! Serde interoperability helpers salvaged across Spodeian projects.

/// Serializes and deserializes unsigned `u64` integers via `i64`-compatible
/// wire representations (e.g., BSON) by bit-casting cleanly across signed/unsigned visitors.
pub mod u64_i64_compat {
    use serde::{Deserializer, Serializer};

    /// Serializes a `u64` as an `i64` for formats like BSON that lack unsigned 64-bit types.
    pub fn serialize<S>(val: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_i64(*val as i64)
    }

    /// Deserializes either an `i64` or `u64` back into `u64`.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct U64Visitor;

        impl<'de> serde::de::Visitor<'de> for U64Visitor {
            type Value = u64;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an integer (i64 or u64)")
            }

            fn visit_i64<E>(self, value: i64) -> Result<u64, E>
            where
                E: serde::de::Error,
            {
                Ok(value as u64)
            }

            fn visit_u64<E>(self, value: u64) -> Result<u64, E>
            where
                E: serde::de::Error,
            {
                Ok(value)
            }
        }

        deserializer.deserialize_any(U64Visitor)
    }
}

#[cfg(all(test, feature = "bson"))]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct HighBitCounter {
        #[serde(with = "super::u64_i64_compat")]
        value: u64,
    }

    #[test]
    fn test_u64_i64_bson_roundtrip() {
        for value in [0_u64, 42, i64::MAX as u64, u64::MAX - 1, u64::MAX] {
            let item = HighBitCounter { value };
            let encoded = crate::export_to_compressed_bson(&item).expect("bson encode");
            let decoded: HighBitCounter =
                crate::import_from_compressed_bson(&encoded).expect("bson decode");
            assert_eq!(decoded, item);
        }
    }
}

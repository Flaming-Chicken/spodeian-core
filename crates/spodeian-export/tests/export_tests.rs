use serde::{Deserialize, Serialize};
use spodeian_export::{
    escape_csv_field, export_to_compressed_bson, export_to_json, import_from_compressed_bson,
    import_from_json, parse_csv_line,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct TestRecord {
    id: u64,
    title: String,
    description: String,
    active: bool,
}

#[test]
fn test_csv_field_escaping() {
    assert_eq!(escape_csv_field("simple"), "simple");
    assert_eq!(escape_csv_field("with,comma"), "\"with,comma\"");
    assert_eq!(escape_csv_field("with\"quote"), "\"with\"\"quote\"");
    assert_eq!(escape_csv_field("with\nnewline"), "\"with\nnewline\"");
}

#[test]
fn test_csv_line_parsing() {
    let line = "1,\"Title, with comma\",\"Desc with \"\"quotes\"\"\",true";
    let fields = parse_csv_line(line);
    assert_eq!(fields.len(), 4);
    assert_eq!(fields[0], "1");
    assert_eq!(fields[1], "Title, with comma");
    assert_eq!(fields[2], "Desc with \"quotes\"");
    assert_eq!(fields[3], "true");
}

#[test]
fn test_json_roundtrip() {
    let record = TestRecord {
        id: 42,
        title: "Alpha".to_string(),
        description: "Benchmark item".to_string(),
        active: true,
    };
    let json_str = export_to_json(&record).expect("Failed to export json");
    let restored: TestRecord = import_from_json(&json_str).expect("Failed to import json");
    assert_eq!(record, restored);
}

#[test]
fn test_compressed_bson_roundtrip() {
    let record = TestRecord {
        id: 100,
        title: "Beta".to_string(),
        description: "Compressed BSON entity".to_string(),
        active: false,
    };
    let compressed_bytes = export_to_compressed_bson(&record).expect("Failed to export bson");
    assert!(!compressed_bytes.is_empty());
    let restored: TestRecord =
        import_from_compressed_bson(&compressed_bytes).expect("Failed to import bson");
    assert_eq!(record, restored);
}

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};
use spodeian_export::{
    export_to_compressed_bson, export_to_csv, export_to_json, import_from_compressed_bson,
    import_from_json, parse_csv_table,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchItem {
    id: String,
    title: String,
    description: String,
    priority: String,
    completed: bool,
    created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchState {
    items: Vec<BenchItem>,
}

fn build_sample_data() -> (BenchState, Vec<Vec<String>>) {
    let mut items = Vec::with_capacity(100);
    let mut rows = Vec::with_capacity(100);
    for i in 0..100 {
        let id = format!("item-{i}");
        let title = format!("Item Title {i}");
        let description = format!("Detailed description for item number {i}, with \"quotes\"");
        items.push(BenchItem {
            id: id.clone(),
            title: title.clone(),
            description: description.clone(),
            priority: "Medium".to_string(),
            completed: i % 2 == 0,
            created_at: 1_700_000_000 + i as u64,
        });
        rows.push(vec![
            id,
            title,
            description,
            "Medium".to_string(),
            (i % 2 == 0).to_string(),
            (1_700_000_000 + i as u64).to_string(),
        ]);
    }
    (BenchState { items }, rows)
}

fn bench_serialization(c: &mut Criterion) {
    let (state, rows) = build_sample_data();
    let headers = [
        "id",
        "title",
        "description",
        "priority",
        "completed",
        "created_at",
    ];

    let json_sample = export_to_json(&state).expect("json export");
    let csv_sample = export_to_csv(&headers, &rows);
    let bson_sample = export_to_compressed_bson(&state).expect("bson export");

    c.bench_function("export_to_json_100_items", |b| {
        b.iter(|| {
            let res = export_to_json(black_box(&state));
            let _ = black_box(res);
        })
    });

    c.bench_function("import_from_json_100_items", |b| {
        b.iter(|| {
            let res: Result<BenchState, _> = import_from_json(black_box(&json_sample));
            let _ = black_box(res);
        })
    });

    c.bench_function("export_to_csv_100_items", |b| {
        b.iter(|| {
            let res = export_to_csv(black_box(&headers), black_box(&rows));
            let _ = black_box(res);
        })
    });

    c.bench_function("parse_csv_table_100_items", |b| {
        b.iter(|| {
            let res = parse_csv_table(black_box(&csv_sample));
            let _ = black_box(res);
        })
    });

    c.bench_function("export_to_compressed_bson_100_items", |b| {
        b.iter(|| {
            let res = export_to_compressed_bson(black_box(&state));
            let _ = black_box(res);
        })
    });

    c.bench_function("import_from_compressed_bson_100_items", |b| {
        b.iter(|| {
            let res: Result<BenchState, _> = import_from_compressed_bson(black_box(&bson_sample));
            let _ = black_box(res);
        })
    });
}

criterion_group!(benches, bench_serialization);
criterion_main!(benches);

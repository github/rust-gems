use criterion::{criterion_group, criterion_main, Criterion};
use github_pspack::{psstruct, RepeatedField};

psstruct!(
    struct LocationData {
        some_field: bool,
        another_field: &str,
        yet_another_field: i32,
    }
);

psstruct!(
    struct Metadata {
        path: &str,
        repo_name: &str,
        doc_id: u32,
        timestamps: RepeatedField<u32>,
        location: LocationData,
        locations: RepeatedField<LocationData<'a>>,
    }
);

fn build_nested_struct() -> Vec<u8> {
    let mut buffer = Vec::new();
    {
        let mut locbuffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut locbuffer);
            let mut builder = LocationDataBuilder::new(writer);
            builder.some_field(true).unwrap();
            builder.another_field("asdf").unwrap();
            builder.yet_another_field(-850).unwrap();
            builder.finish().unwrap();
        }

        let loc = LocationData::from_bytes(&locbuffer);

        let writer = std::io::Cursor::new(&mut buffer);
        let mut builder = MetadataBuilder::new(writer);
        builder.location(loc).unwrap();
        builder.finish().unwrap();
    }
    buffer
}

#[allow(dead_code)]
fn bench_struct_reading(c: &mut Criterion) {
    let input = build_nested_struct();
    c.bench_function("struct reading", |b| {
        b.iter(|| {
            let meta = Metadata::from_bytes(&input);
            let loc = meta.location();
            loc.yet_another_field()
        });
    });
}

criterion_group!(benches, bench_struct_reading);
criterion_main!(benches);

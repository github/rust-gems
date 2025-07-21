use geo_filters::build_hasher::UnstableDefaultBuildHasher;

// Mirrors what we have in the README.
// If this breaks then it should be fixed an the readme updated.
#[test]
fn readme() {
    use geo_filters::distinct_count::GeoDistinctCount13;
    use geo_filters::Count;

    let mut c1 = GeoDistinctCount13::default();
    c1.push(1);
    c1.push(2);

    let mut c2 = GeoDistinctCount13::default();
    c2.push(2);
    c2.push(3);

    let estimated_size = c1.size_with_sketch(&c2);
    assert!((3.0_f32 * 0.9..=3.0_f32 * 1.1).contains(&estimated_size));
}

#[test]
fn can_use_predefined_diff_count() {
    use geo_filters::diff_count::GeoDiffCount7;
    use geo_filters::Count;
    let mut f = GeoDiffCount7::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_custom_diff_count() {
    use geo_filters::diff_count::{GeoDiffConfig7, GeoDiffCount};
    use geo_filters::Count;
    let mut f = GeoDiffCount::<GeoDiffConfig7>::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_predefined_config_value() {
    use geo_filters::diff_count::{GeoDiffConfig7, GeoDiffCount};
    use geo_filters::Count;
    let c = GeoDiffConfig7::<UnstableDefaultBuildHasher>::default();
    let mut f = GeoDiffCount::new(c);
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_fixed_config_value() {
    use geo_filters::config::FixedConfig;
    use geo_filters::diff_count::GeoDiffCount;
    use geo_filters::Count;
    let c = FixedConfig::<_, u16, 7, 128, 10, UnstableDefaultBuildHasher>::default();
    let mut f = GeoDiffCount::new(c);
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_variable_config_value() {
    use geo_filters::config::VariableConfig;
    use geo_filters::diff_count::GeoDiffCount;
    use geo_filters::Count;
    let c = VariableConfig::<_, u16, UnstableDefaultBuildHasher>::new(7, 128, 10);
    let mut f = GeoDiffCount::new(c);
    f.push(42);
    f.size();
}

#[test]
fn can_use_predefined_distinct_count() {
    use geo_filters::distinct_count::GeoDistinctCount7;
    use geo_filters::Count;
    let mut f = GeoDistinctCount7::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_custom_distinct_count() {
    use geo_filters::distinct_count::{GeoDistinctConfig7, GeoDistinctCount};
    use geo_filters::Count;
    let mut f = GeoDistinctCount::<GeoDistinctConfig7>::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_predefined_config_value() {
    use geo_filters::distinct_count::{GeoDistinctConfig7, GeoDistinctCount};
    use geo_filters::Count;
    let c = GeoDistinctConfig7::<UnstableDefaultBuildHasher>::default();
    let mut f = GeoDistinctCount::new(c);
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_fixed_config_value() {
    use geo_filters::config::FixedConfig;
    use geo_filters::distinct_count::GeoDistinctCount;
    use geo_filters::Count;
    let c = FixedConfig::<_, u16, 7, 118, 11, UnstableDefaultBuildHasher>::default();
    let mut f = GeoDistinctCount::new(c);
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_variable_config_value() {
    use geo_filters::config::VariableConfig;
    use geo_filters::distinct_count::GeoDistinctCount;
    use geo_filters::Count;
    let c = VariableConfig::<_, u16, UnstableDefaultBuildHasher>::new(7, 118, 11);
    let mut f = GeoDistinctCount::new(c);
    f.push(42);
    f.size();
}

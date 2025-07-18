use fnv::FnvBuildHasher;

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
    let mut f = GeoDiffCount::<GeoDiffConfig7, FnvBuildHasher>::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_predefined_config_value() {
    use geo_filters::diff_count::{GeoDiffConfig7, GeoDiffCount};
    use geo_filters::Count;
    let c = GeoDiffConfig7::default();
    let mut f = GeoDiffCount::new(c, FnvBuildHasher::default());
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_fixed_config_value() {
    use geo_filters::config::FixedConfig;
    use geo_filters::diff_count::GeoDiffCount;
    use geo_filters::Count;
    let c = FixedConfig::<_, u16, 7, 128, 10>::default();
    let mut f = GeoDiffCount::new(c, FnvBuildHasher::default());
    f.push(42);
    f.size();
}

#[test]
fn can_use_diff_count_with_variable_config_value() {
    use geo_filters::config::VariableConfig;
    use geo_filters::diff_count::GeoDiffCount;
    use geo_filters::Count;
    let c = VariableConfig::<_, u16>::new(7, 128, 10);
    let mut f = GeoDiffCount::new(c, FnvBuildHasher::default());
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
    let mut f = GeoDistinctCount::<GeoDistinctConfig7, FnvBuildHasher>::default();
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_predefined_config_value() {
    use geo_filters::distinct_count::{GeoDistinctConfig7, GeoDistinctCount};
    use geo_filters::Count;
    let c = GeoDistinctConfig7::default();
    let mut f = GeoDistinctCount::new(c, FnvBuildHasher::default());
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_fixed_config_value() {
    use geo_filters::config::FixedConfig;
    use geo_filters::distinct_count::GeoDistinctCount;
    use geo_filters::Count;
    let c = FixedConfig::<_, u16, 7, 118, 11>::default();
    let mut f = GeoDistinctCount::new(c, FnvBuildHasher::default());
    f.push(42);
    f.size();
}

#[test]
fn can_use_distinct_count_with_variable_config_value() {
    use geo_filters::config::VariableConfig;
    use geo_filters::distinct_count::GeoDistinctCount;
    use geo_filters::Count;
    let c = VariableConfig::<_, u16>::new(7, 118, 11);
    let mut f = GeoDistinctCount::new(c, FnvBuildHasher::default());
    f.push(42);
    f.size();
}

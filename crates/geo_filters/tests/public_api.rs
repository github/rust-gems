use geo_filters::build_hasher::UnstableDefaultBuildHasher;

#[test]
fn custom_method_can_use_standard_lazy_lookups() {
    use geo_filters::config::{FixedConfig, GeoConfig, Lookup, Lookups};
    use geo_filters::{Distinct, Method};
    use std::sync::LazyLock;

    #[derive(Clone, PartialEq, Eq)]
    struct CustomMethod;

    impl Method for CustomMethod {}

    impl Lookups for CustomMethod {
        fn get_lookups() -> &'static [LazyLock<Lookup>] {
            <Distinct as Lookups>::get_lookups()
        }

        fn new_lookup(b: usize) -> Lookup {
            <Distinct as Lookups>::new_lookup(b)
        }
    }

    let config = FixedConfig::<CustomMethod, u16, 7, 136, 8, UnstableDefaultBuildHasher>::default();
    assert_eq!(config.hash_to_bucket(u64::MAX), 0);
    assert_eq!(config.expected_items(0), 0.0);
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

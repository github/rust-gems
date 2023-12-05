//! Lock-free data structure tests.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Barrier};

use crate::*;

#[test]
fn vec_u32() {
    let arena = Arena::new();
    let vec = arena.alloc_default::<AVec<u32>>();
    vec.push(17);
    vec.push(36);
    vec.push(177);

    assert_eq!(vec.len(), 3);
    assert_eq!(vec[2], 177);
    assert_eq!(&vec[..], &[17, 36, 177]);
    assert_eq!(vec.iter().copied().collect::<Vec<u32>>(), vec![17, 36, 177]);
}

#[test]
fn vec_vec_u32() {
    let arena = Arena::new();
    let root = arena.alloc_default::<AVec<AVec<u32>>>();
    root.push_default();

    assert_eq!(root.len(), 1);

    let w0 = root.get_writer(0).unwrap();
    w0.push(1);
    w0.push(2);

    root.push_default();
    let w1 = root.get_writer(1).unwrap();
    w1.push(3);
    w1.push(4);

    assert_eq!(format!("{:?}", root), "[[1, 2], [3, 4]]");

    // Note: You need to know what you are doing when pushing a mutable object into a vector, since
    // this shared object also shares "some" of the mutations until e.g. a vector has to be resized.
    // From a data perspective, this is fine, but it can be confusing.
    root.push(w1);

    assert_eq!(format!("{:?}", root), "[[1, 2], [3, 4], [3, 4]]");

    let w1 = root.get_writer(1).unwrap();
    for i in 5..=11 {
        w1.push(i);
    }

    assert_eq!(
        format!("{:?}", root),
        "[[1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4, 5, 6, 7, 8, 9, 10]]"
    );
}

#[test]
fn vec_growing() {
    let arena = Arena::new();
    let vec = arena.alloc_default::<AVec<u32>>();
    vec.push(0);
    vec.push(1);
    vec.push(2);
    for i in 0..4 {
        // Normal vectors don't allow using an iterator and at the same time
        // mutating the same vector. `AVec` allows this, but it is safe.
        // Growing the vector does not invalidate existing iterators.
        for val in vec.iter() {
            vec.push(*val + 3 * (1 << i));
        }
    }

    let expected = (0..48).collect::<Vec<u32>>();
    assert_eq!(vec.as_slice(), expected.as_slice());
}

#[test]
fn vec_racing() {
    const N: u32 = 1200;
    const NREADERS: usize = 8;

    for _cycle in 0..50 {
        let arena = Arena::new();
        let v = arena.alloc_default::<AVec<u32>>();

        // Try to make all threads start at once, to maximize chances of a race condition
        let barrier = Arc::new(Barrier::new(NREADERS + 1));

        std::thread::scope(|scope| {
            for _i in 0..8 {
                let barrier = Arc::clone(&barrier);
                let v = v.read_handle();
                scope.spawn(move || {
                    barrier.wait();
                    let mut count = 0;
                    while count < N as usize {
                        let result = v.iter().copied().collect::<Vec<u32>>();
                        assert!(result.len() >= count);
                        count = result.len();
                        assert_eq!(result, (0..count as u32).collect::<Vec<u32>>());
                    }
                });
            }

            // writer
            barrier.wait();
            for i in 0..N {
                v.push(i);
                std::thread::yield_now();
            }

            assert_eq!(
                v.iter().copied().collect::<Vec<u32>>(),
                (0..N).collect::<Vec<u32>>()
            );
        });
    }
}

#[test]
fn test_deref() {
    let arena = Arena::new();
    let v = arena.alloc_default::<AVec<i64>>();

    v.push(7);
    v.push(14);
    v.push(217);
    v.push(5432);

    // Writer<AVec<i64>> has all the methods that slices have.
    assert!(v.contains(&5432));

    // So does ReadHandle.
    assert_eq!(v.read_handle().binary_search(&217), Ok(2));

    // So does WriteHandle.
    assert_eq!(v.write_handle().partition_point(|e| *e < 77_000), 4);
}

#[test]
fn test_box() {
    let arena = Arena::new();
    let v = arena.alloc_default::<AVec<ABox<i64>>>();

    let w = arena.alloc(7);
    v.push(w);
    v.push(v.get_writer(0).unwrap());
    v.push(3);

    assert_eq!(format!("{:?}", *v), "[7, 7, 3]");

    v.get_writer(1).unwrap().store(8);

    assert_eq!(format!("{:?}", *v), "[7, 8, 3]");

    v.get_writer(2).unwrap().store(arena.alloc(9));

    assert_eq!(format!("{:?}", *v), "[7, 8, 9]");
}

fn check_is_copy<T: Copy>(_value: T) {}

#[test]
fn test_nesting_of_boxes_and_vecs() {
    let arena = Arena::new();
    let vv = arena.alloc_default::<AVec<ABox<AVec<i64>>>>();
    let v = arena.alloc_default::<AVec<i64>>();
    let w = arena.alloc(1);

    check_is_copy(v);

    v.push(w); // This copies the value 1 into the vector.
    v.push(2);
    vv.push(v);
    vv.push(arena.alloc_default::<AVec<i64>>());

    assert_eq!(format!("{:?}", *vv), "[[1, 2], []]");

    vv.get_writer(1).unwrap().store(v);

    assert_eq!(format!("{:?}", *vv), "[[1, 2], [1, 2]]");

    vv.get_writer(1).unwrap().get_writer().push(3);
    vv.get_writer(1).unwrap().get_writer().push(4);
    vv.get_writer(1).unwrap().get_writer().push(5);

    assert_eq!(format!("{:?}", *vv), "[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]");

    // Check whether we can pass a `Writer<ABox>` to `Writer<ABox>::store`.
    vv.get_writer(1).unwrap().store(vv.get_writer(0).unwrap());
}

#[test]
fn test_ptr() {
    let arena = Arena::new();
    let v = arena.alloc_default::<AVec<Ptr<i64>>>();

    let w = arena.alloc(7);
    v.push(Some(w));
    v.push_default();
    v.push(v.get_writer(1).unwrap());
    v.push(3);

    assert_eq!(format!("{:?}", *v), "[Some(7), None, None, Some(3)]");

    v.get_writer(1).unwrap().store(Some(w));

    assert_eq!(format!("{:?}", *v), "[Some(7), Some(7), None, Some(3)]");

    v.get_writer(2).unwrap().store(Some(arena.alloc(8)));

    assert_eq!(format!("{:?}", *v), "[Some(7), Some(7), Some(8), Some(3)]");
}

#[test]
fn test_nesting_of_ptrs_and_vecs() {
    let arena = Arena::new();
    let vv = arena.alloc_default::<AVec<Ptr<AVec<i64>>>>();
    let v = arena.alloc_default::<AVec<i64>>();
    let w = arena.alloc(1);

    v.push(w); // This copies the value 1 into the vector.
    v.push(2);
    vv.push(Some(v));
    vv.push_default();

    assert_eq!(format!("{:?}", *vv), "[Some([1, 2]), None]");

    vv.get_writer(1).unwrap().store(Some(v));

    assert_eq!(format!("{:?}", *vv), "[Some([1, 2]), Some([1, 2])]");

    vv.get_writer(1).unwrap().get_writer().unwrap().push(3);
    vv.get_writer(1).unwrap().get_writer().unwrap().push(4);
    vv.get_writer(1).unwrap().get_writer().unwrap().push(5);

    assert_eq!(
        format!("{:?}", *vv),
        "[Some([1, 2, 3, 4, 5]), Some([1, 2, 3, 4, 5])]"
    );

    vv.get_writer(0).unwrap().clear();

    assert_eq!(format!("{:?}", *vv), "[None, Some([1, 2, 3, 4, 5])]");

    // Check whether we can assign an optional writer of a ptr to another ptr.
    vv.get_writer(1).unwrap().store(vv.get_writer(0).unwrap());
}

#[test]
fn hash_u32() {
    let arena = Arena::new();
    let map = arena.alloc_default::<AHashMap<AtomicU32, usize>>();

    const P: u32 = 7193;
    let mut x = 1u32;
    for _ in 0..P - 1 {
        let y = x * 308 % P;
        assert_eq!(*map.get_or_insert(y, x as usize), x as usize);
        x = y;
    }

    let mut key_seen = vec![false; P as usize];
    let mut value_seen = vec![false; P as usize];
    for (k, v) in map.iter() {
        key_seen[k.load(Ordering::Relaxed) as usize] = true;
        value_seen[*v] = true;
    }
    assert!(!key_seen[0]);
    assert!((1..P as usize).all(|i| key_seen[i]));
    assert!(!value_seen[0]);
    assert!((1..P as usize).all(|i| value_seen[i]));
}

#[test]
fn hash_vec() {
    let arena = Arena::new();
    let data: [(u32, &str); 6] = [
        (1, "one"),
        (2, "two"),
        (1, "ichi"),
        (2, "ni"),
        (1, "un"),
        (2, "deux"),
    ];

    let map = arena.alloc_default::<AHashMap<AtomicU32, AVec<&'static str>>>();
    for (k, v) in data {
        map.get_or_insert_default(k).push(v);
    }

    assert_eq!(map.get(1).unwrap().as_slice(), &["one", "ichi", "un"]);
    assert_eq!(map.get(2).unwrap().as_slice(), &["two", "ni", "deux"]);

    if let Some(wv) = map.get_writer(1) {
        wv.push("aik");
    }
    assert_eq!(map.get(1).unwrap().join(", "), "one, ichi, un, aik");
}

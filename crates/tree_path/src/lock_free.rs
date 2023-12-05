use crate::TreePath;

use github_lock_free::{
    serializable::APackBuffer,
    traits::{IntoArena, SelfContainedDefault},
    Arena, CloneWithinArena, Ptr,
};

#[derive(Default)]
pub struct ATreePath {
    ptr: Ptr<APackBuffer>,
}

impl ATreePath {
    pub fn get(&self) -> TreePath {
        TreePath::from_bytes(self.as_slice())
    }

    fn as_slice(&self) -> &[u8] {
        self.ptr.get().map(|x| x.as_slice()).unwrap_or_default()
    }
}

impl CloneWithinArena for ATreePath {
    unsafe fn clone_within_arena(&self) -> Self {
        Self {
            ptr: unsafe { self.ptr.clone_within_arena() },
        }
    }
}

impl SelfContainedDefault for ATreePath {}

impl<'a> IntoArena<ATreePath> for TreePath<'a> {
    unsafe fn into_arena(self, arena: &Arena) -> ATreePath {
        let ptr = APackBuffer::allocate(arena, self.as_bytes());
        let abox = unsafe { ptr.into_arena(arena) };
        ATreePath { ptr: abox.into() }
    }
}

impl std::fmt::Debug for ATreePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get())
    }
}

#[cfg(test)]
mod tests {
    //! Lock-free data structure tests.

    use github_lock_free::serializable::ASerialized;
    use github_lock_free::*;

    use crate::lock_free::*;
    use crate::*;

    #[test]
    fn test_tree_path() {
        let arena = Arena::new();
        let vec = arena.alloc_default::<AVec<ATreePath>>();
        vec.push_default();
        vec.push(arena.alloc_default::<ATreePath>());
        vec.push(TreePath::from_ids(&[1]));

        assert_eq!(format!("{vec:?}"), "[ (/),  (/), 00000001 (/1)]");
    }

    #[test]
    fn hash_tree_path() {
        #[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
        struct SnapshotInfo {
            name: &'static str,
        }

        impl CloneWithinArena for SnapshotInfo {
            unsafe fn clone_within_arena(&self) -> Self {
                *self
            }
        }

        let arena = Arena::new();
        let map = arena.alloc_default::<AHashMap<ASerialized<TreePathFormat>, SnapshotInfo>>();
        map.get_or_insert_default(TreePath::from_ids(&[6, 1, OO, OO, OO, OO]));
        map.get_or_insert_with(TreePath::from_ids(&[7]), || SnapshotInfo { name: "flynn" });
        map.get_or_insert_default(TreePath::from_ids(&[6, 1, OO, OO, OO, OO]));
        map.get_or_insert_default(TreePath::from_ids(&[6, 1, OO, OO, OO, OO, OO]));
        map.get_or_insert_default(TreePath::from_ids(&[6, 1, OO, OO, OO, OO]));
        map.get_or_insert_with(TreePath::from_ids(&[6, 1, OO, OO, OO, OO, OO]), || {
            panic!("should not be called")
        });

        assert_eq!(
            map.get(TreePath::from_ids(&[6, 1, OO, OO, OO, OO])),
            Some(&SnapshotInfo::default())
        );
        assert_eq!(map.get(TreePath::from_ids(&[7])).unwrap().name, "flynn");
    }

    #[test]
    fn test_serializable() {
        // Now we can use the new dummy type in our lock free setup.
        let arena = Arena::new();
        let vec = arena.alloc_default::<AVec<ASerialized<TreePathFormat>>>();
        vec.push_default();
        vec.push(arena.alloc_default::<ASerialized<TreePathFormat>>());
        vec.push(TreePath::from_ids(&[1]));

        assert_eq!(format!("{vec:?}"), "[ (/),  (/), 00000001 (/1)]");
    }
}

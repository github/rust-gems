#[macro_export]
macro_rules! psstruct {
    (
        $( # [ $attrs:meta ] )*
        $vis:vis struct $name:ident {
            $( $field_name:ident: $type:ty, )*
        }
    ) => {
        $crate::paste!{
            $( # [ $attrs ] )*
            $vis struct $name<'a> {
                inner: $crate::PSStruct<'a>,
            }

            #[allow(non_camel_case_types)]
            enum [<$name FieldId>] { $($field_name,)* }

            $vis struct [<$name Builder>]<W: std::io::Write> {
                inner: $crate::PSStructBuilder<W>,
                last_field_number: Option<usize>,
            }

            impl<W: std::io::Write> [<$name Builder>]<W> {
                #[allow(dead_code)]
                $vis fn new(writer: W) -> Self {
                    Self {
                        inner: $crate::PSStructBuilder::new(writer),
                        last_field_number: None
                    }
                }

                #[allow(dead_code)]
                $vis fn finish(self) -> std::io::Result<usize> {
                    self.inner.finish()
                }

                $(
                    $crate::psbuilderfield!{$vis $field_name, $type, [<$name FieldId>]::$field_name as usize}
                )*
            }

            impl<'a> $name<'a> {
                $vis fn from_bytes(buf: &'a [u8]) -> Self {
                    Self {
                        inner: $crate::PSStruct::from_bytes(buf)
                    }
                }

                #[allow(dead_code)]
                $vis fn buffer(&self) -> &'a [u8] {
                    self.inner.content
                }

                $(
                    $crate::psfield!{$vis $field_name, $type, [<$name FieldId>]::$field_name as u32}
                )*
            }

            impl<'a> $crate::Serializable<'a> for $name<'a> {
                fn from_bytes(buffer: &'a [u8]) -> Self {
                    Self::from_bytes(buffer)
                }

                fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<usize> {
                    writer.write_all(self.inner.content)?;
                    Ok(self.inner.content.len())
                }
            }
        }
    };
}

#[macro_export]
macro_rules! psbuilderfield {
    ($vis:vis $field_name:ident, RepeatedField<&$type:ty>, $field_number:expr) => {
        $vis fn $field_name<'a, 'b, T: IntoIterator<Item = &'b &'b $type>>(
            &'b mut self,
            iter: T,
        ) -> std::io::Result<()> {
            let current_field_number = if let Some(x) = self.last_field_number {
                if x >= $field_number {
                    panic!("must write fields in order!");
                }
                x + 1
            } else {
                0
            };

            for _ in current_field_number..$field_number {
                // Need to skip any fields not yet sent
                self.inner.skip()?;
            }

            self.last_field_number = Some($field_number);

            let writer = &mut self.inner.writer;
            let size = {
                let mut builder = RepeatedFieldBuilder::new(writer);
                for value in iter {
                    builder.push(value)?;
                }
                builder.finish()?
            };

            self.inner.sizes.push(size as u32);
            Ok(())
        }
    };
    ($vis:vis $field_name:ident, RepeatedField<$type:ty>, $field_number:expr) => {
        $crate::paste! {
            #[allow(dead_code)]
            $vis fn [<mut_ $field_name>]<'a>(&'a mut self) -> $crate::RepeatedFieldBuilderWrapper<$type, &'a mut W> where &'a mut W: std::io::Write {
                let current_field_number = if let Some(x) = self.last_field_number {
                    if x >= $field_number {
                        panic!("must write fields in order!");
                    }
                    x + 1
                } else {
                    0
                };

                for _ in current_field_number..$field_number {
                    // Need to skip any fields not yet sent
                    self.inner.skip().expect("failed to write skip field");
                }

                self.last_field_number = Some($field_number);

                $crate::RepeatedFieldBuilderWrapper::new(
                    &mut self.inner.sizes,
                    $crate::RepeatedFieldBuilder::new(&mut self.inner.writer),
                )
            }
        }

        #[allow(dead_code)]
        pub fn $field_name<'a, T: IntoIterator<Item = &'a $type>>(
            &mut self,
            iter: T,
        ) -> std::io::Result<()> {
            let current_field_number = if let Some(x) = self.last_field_number {
                if x >= $field_number {
                    panic!("must write fields in order!");
                }
                x + 1
            } else {
                0
            };

            for _ in current_field_number..$field_number {
                // Need to skip any fields not yet sent
                self.inner.skip()?;
            }

            self.last_field_number = Some($field_number);

            let writer = &mut self.inner.writer;
            let size = {
                let mut builder = $crate::RepeatedFieldBuilder::new(writer);
                for value in iter {
                    builder.push(value)?;
                }
                builder.finish()?
            };

            self.inner.sizes.push(size as u32);
            Ok(())
        }
    };
    ($vis:vis $field_name:ident, $type:ty, $field_number:expr) => {
        #[allow(dead_code, clippy::extra_unused_lifetimes)]
        $vis fn $field_name<'a>(&mut self, value: $type) -> std::io::Result<()> {
            let current_field_number = if let Some(x) = self.last_field_number {
                if x >= $field_number {
                    panic!("must write fields in order!");
                }
                x + 1
            } else {
                0
            };

            for _ in current_field_number..$field_number {
                // Need to skip any fields not yet sent
                self.inner.skip()?;
            }

            self.last_field_number = Some($field_number);
            self.inner.push(value)
        }
    };
}

#[macro_export]
macro_rules! psfield {
    ($vis:vis $name:ident, &$type:ty, $field_number:expr) => {
        #[allow(dead_code)]
        $vis fn $name(&self) -> &'a $type {
            self.inner.read($field_number)
        }
    };
    ($vis:vis $name:ident, RepeatedField<&$type:ty>, $field_number:expr) => {
        #[allow(dead_code)]
        $vis fn $name(&self) -> RepeatedField<'a, &'a $type> {
            self.inner.read_repeated($field_number)
        }
    };
    ($vis:vis $name:ident, RepeatedField<$type:ty>, $field_number:expr) => {
        #[allow(dead_code)]
        $vis fn $name(&self) -> RepeatedField<'a, $type> {
            self.inner.read_repeated($field_number)
        }
    };
    ($vis:vis $name:ident, $type:ty, $field_number:expr) => {
        #[allow(dead_code, clippy::needless_lifetimes)]
        // TODO: The 'b lifetime should not be here, but is needed to make RepeatedFields happy.
        // This probably related to a more fundamental lifetime problems with RepeatedFields and these macros.
        $vis fn $name<'b>(&'b self) -> $type {
            self.inner.read($field_number)
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{RepeatedField, RepeatedFieldBuilder};

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

    #[test]
    fn test_struct_sigsev() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = MetadataBuilder::new(writer);
            builder.path("/x/y/z.txt").unwrap();
            builder.repo_name("github/blackbird").unwrap();
            builder.doc_id(35).unwrap();
            builder.finish().unwrap();
        }

        // Test lifetime of returned path string. It should outlive the reader instance and have
        // the lifetime of the underlying buffer.
        let path;
        {
            let meta = Metadata::from_bytes(&buffer);
            path = meta.path();
            assert_eq!(meta.path(), "/x/y/z.txt");
            assert_eq!(meta.doc_id(), 35);
            assert_eq!(meta.repo_name(), "github/blackbird");
        }
        assert_eq!(path, "/x/y/z.txt");
    }

    #[test]
    fn test_struct_omit_field() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = MetadataBuilder::new(writer);

            // It's OK to skip the other fields and just write doc ID. We'll
            // fill in skip markers so that it can still be read correctly
            builder.doc_id(35).unwrap();
            builder.finish().unwrap();
        }

        let meta = Metadata::from_bytes(&buffer);
        assert_eq!(meta.path(), "");
        assert_eq!(meta.doc_id(), 35);
        assert_eq!(meta.repo_name(), "");
    }

    #[test]
    #[should_panic]
    fn test_out_of_order_write() {
        let mut buffer = Vec::new();
        let writer = std::io::Cursor::new(&mut buffer);
        let mut builder = MetadataBuilder::new(writer);
        builder.repo_name("github/blackbird").unwrap();
        builder.path("/x/y/z.txt").unwrap();
    }

    #[test]
    fn test_repeated_field_access() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = MetadataBuilder::new(writer);
            builder.timestamps(&[2, 4, 6, 8]).unwrap();
            builder.finish().unwrap();
        }

        let meta = Metadata::from_bytes(&buffer);
        let ts = meta.timestamps();

        assert_eq!(ts.get(0), 2);
        assert_eq!(ts.get(1), 4);
        assert_eq!(ts.get(2), 6);
        assert_eq!(ts.get(3), 8);
    }

    #[test]
    fn test_repeated_field_iteration() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = MetadataBuilder::new(writer);
            builder.timestamps(&[2, 4, 6, 8]).unwrap();
            builder.finish().unwrap();
        }

        let meta = Metadata::from_bytes(&buffer);
        let ts = meta.timestamps();
        let mut iter = ts.into_iter();

        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 6);
        assert_eq!(iter.next().unwrap(), 8);
        assert_eq!(iter.next(), None);
    }

    psstruct!(
        struct Terms {
            terms: RepeatedField<&str>,
        }
    );

    #[test]
    fn test_repeated_str() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = TermsBuilder::new(writer);
            builder.terms(&["test", "a", "b", "c"]).unwrap();
            builder.finish().unwrap();
        }

        let terms = Terms::from_bytes(&buffer);
        let ts = terms.terms();
        let output = ts.into_iter().collect::<Vec<_>>();

        assert_eq!(&output, &["test", "a", "b", "c"]);
    }

    fn ch(c: char) -> u8 {
        c as u8
    }

    #[test]
    fn test_repeated_struct() {
        let mut buffer = Vec::new();
        {
            let writer = std::io::Cursor::new(&mut buffer);
            let mut builder = MetadataBuilder::new(writer);
            {
                let mut locs = builder.mut_locations();
                for idx in 0..10 {
                    let mut locbuffer = Vec::new();
                    {
                        let writer = std::io::Cursor::new(&mut locbuffer);
                        let mut builder = LocationDataBuilder::new(writer);
                        builder.some_field(true).unwrap();
                        builder.another_field("asdf").unwrap();
                        builder.yet_another_field(-12 * idx).unwrap();
                        builder.finish().unwrap();
                    }

                    locs.push_bytes(&locbuffer).unwrap();
                }
            }
            builder.finish().unwrap();
        }

        let meta = Metadata::from_bytes(&buffer);
        let locs = meta.locations();
        assert_eq!(locs.iter().count(), 10);
        assert_eq!(
            locs.iter()
                .map(|x| x.yet_another_field())
                .collect::<Vec<_>>(),
            vec![0, -12, -24, -36, -48, -60, -72, -84, -96, -108]
        );
    }

    #[test]
    fn test_builder_nested_struct() {
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

        // Inspect the layout
        assert_eq!(
            &buffer,
            &[
                1, // bool field
                ch('a'),
                ch('s'),
                ch('d'),
                ch('f'),
                174,
                252,
                255,
                255,
                1,  // End of some_field
                5,  // End of another_field
                9,  // End of yet_another_field
                3,  // Number of fields in LocationData
                0,  // Skip marker
                0,  // Skip marker
                0,  // Skip marker
                0,  // Skip marker
                13, // End of location field
                5,  // Number of fields in Metadata
            ],
        );

        let meta = Metadata::from_bytes(&buffer);
        let loc = meta.location();
        assert!(loc.some_field());
        assert_eq!(loc.another_field(), "asdf");
        assert_eq!(loc.yet_another_field(), -850);
    }
}

use std::io::Read;

use arbitrary::{Arbitrary, Unstructured};
use github_pspack::RepeatedField;

#[derive(Debug, Default, PartialEq, Arbitrary)]
struct DecodedFormat {
    a: u8,
    b: i32,
    c: u32,
    d: bool,
    e: String,
    children: Vec<DecodedFormat>,
}

github_pspack::psstruct!(
    struct EncodedFormat {
        a: u8,
        b: i32,
        c: u32,
        d: bool,
        e: &str,
        children: RepeatedField<EncodedFormat<'a>>,
    }
);

fn encode_struct(d: &DecodedFormat) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let writer = std::io::Cursor::new(&mut buf);
        let mut builder = EncodedFormatBuilder::new(writer);
        builder.a(d.a).expect("couldn't read a");
        builder.b(d.b).expect("couldn't read b");
        builder.c(d.c).expect("couldn't read c");
        builder.d(d.d).expect("couldn't read d");
        builder.e(&d.e).expect("couldn't read e");

        {
            let mut children = builder.mut_children();
            for child in &d.children {
                children
                    .push_bytes(&encode_struct(child))
                    .expect("couldn't encode child");
            }
        }

        builder.finish().expect("couldn't finish struct");
    }
    buf
}

fn decode_struct(encoded: &EncodedFormat) -> DecodedFormat {
    let decoded = DecodedFormat {
        a: encoded.a(),
        b: encoded.b(),
        c: encoded.c(),
        d: encoded.d(),
        e: encoded.e().to_owned(),
        children: encoded
            .children()
            .into_iter()
            .map(|e| decode_struct(&e))
            .collect(),
    };

    decoded
}

fn main() {
    let mut buf = Vec::new();
    std::io::stdin()
        .read_to_end(&mut buf)
        .expect("couldn't read stdin");
    let mut unstructured = Unstructured::new(&buf);

    let decoded = match DecodedFormat::arbitrary(&mut unstructured) {
        Err(err) => {
            // usually just "not enough bytes"
            println!("{}", err);
            return;
        }
        Ok(value) => value,
    };
    let encoded = encode_struct(&decoded);

    let read = decode_struct(&EncodedFormat::from_bytes(&encoded));

    if read != decoded {
        eprintln!("expected {:#?}, got {:#?}", decoded, read);
        std::process::exit(1);
    }
}

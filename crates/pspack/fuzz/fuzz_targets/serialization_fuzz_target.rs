#![no_main]
use libfuzzer_sys::fuzz_target;
use github_pspack::RepeatedField;

#[derive(Debug, Default, PartialEq)]
struct DecodedFormat {
    a: u8,
    b: i32,
    c: u32,
    d: bool,
    e: String,
    children: Vec<DecodedFormat>,
}

pspack::psstruct!(
    EncodedFormat,
    a: u8,
    b: i32,
    c: u32,
    d: bool,
    e: &str,
    children: RepeatedField<EncodedFormat<'a>>,
);

impl DecodedFormat {
    fn from_bytes(mut buf: &[u8]) -> Self {
        let mut decoded = DecodedFormat::default();
        if !buf.is_empty() {
            decoded.a = buf[0];
            buf = &buf[1..];
        }

        if buf.len() >= 4 {
            decoded.b = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            buf = &buf[4..];
        }

        if buf.len() >= 4 {
            decoded.c = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            buf = &buf[4..];
        }

        if !buf.is_empty() {
            decoded.d = buf[0] > 128;
            buf = &buf[1..];
        }

        if buf.len() < 2 {
            return decoded;
        }

        let str_size = std::cmp::min(buf.len() - 2, u16::from_le_bytes([buf[0], buf[1]]) as usize);
        buf = &buf[2..];

        // Try to decode those bytes as a string (might fail if not UTF-8)
        if let Ok(s) = String::from_utf8(buf[0..str_size].to_owned()) {
            decoded.e = s;
            buf = &buf[str_size..];
        }

        if buf.len() < 2 {
            return decoded;
        }

        // Try to recursively decode children
        let child_size = std::cmp::max(1, u16::from_le_bytes([buf[0], buf[1]]) as usize);
        buf = &buf[2..];

        while !buf.is_empty() {
            let chunk_size = std::cmp::min(buf.len(), child_size);

            decoded
                .children
                .push(DecodedFormat::from_bytes(&buf[0..chunk_size]));
            buf = &buf[chunk_size..];
        }

        decoded
    }
}

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


fuzz_target!(|data: &[u8]| {
    let decoded = DecodedFormat::from_bytes(data);
    let encoded = encode_struct(&decoded);

    let read = decode_struct(&EncodedFormat::from_bytes(&encoded));

    if read != decoded {
        panic!("expected {:#?}, got {:#?}", decoded, read);
    }
});

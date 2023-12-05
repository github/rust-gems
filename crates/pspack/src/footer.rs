pub fn required_space(value: usize) -> usize {
    if (value & !0x3f) == 0 {
        1
    } else if (value & !0x3fff) == 0 {
        2
    } else if (value & !0x3fffff) == 0 {
        3
    } else if (value & !0x3fffffff) == 0 {
        4
    } else {
        // If we want to encode larger values than 1 billion, then we can
        // encode 1 bytes with 0, 2 bytes with 1, 4 bytes with 2, and 8 bytes with 3.
        // Since we will use this format only for encoding buffer sizes, the
        // resulting overhead is completely negligible.
        panic!("value {value} too large for encoding");
    }
}

pub fn write<W: std::io::Write>(mut writer: W, value: usize) -> std::io::Result<usize> {
    let size = required_space(value);
    let mut buf = vec![0; size];
    for (i, item) in buf.iter_mut().enumerate().take(size) {
        *item = (value >> (i * 8)) as u8;
    }
    buf[size - 1] |= ((size - 1) << 6) as u8;
    writer.write_all(&buf)?;
    Ok(size)
}

pub fn read(buf: &[u8]) -> std::io::Result<(usize, usize)> {
    let size = (buf[buf.len() - 1] >> 6) as usize + 1;
    let mut value = 0usize;
    for i in 0..size {
        value += (buf[buf.len() - size + i] as usize) << (i * 8);
    }
    // mask out the 2 bits with the size information.
    value &= !(!0usize << (size * 8 - 2));
    Ok((value, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_write() {
        for i in 0..16000 {
            let mut buffer = Vec::new();
            {
                let mut writer = std::io::Cursor::new(&mut buffer);
                write(&mut writer, i).unwrap();
            }

            let (value, size) = read(&buffer).unwrap();
            assert_eq!(value, i);
            assert_eq!(size, required_space(value));
            assert_eq!(size, buffer.len());
        }
    }
}

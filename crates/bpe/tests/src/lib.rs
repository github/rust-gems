use bpe::byte_pair_encoding::BytePairEncoding;
use rand::{thread_rng, Rng};

pub fn create_test_bytes(bpe: &BytePairEncoding, tokens: usize) -> Vec<u8> {
    let mut text = vec![];
    for _ in 0..tokens {
        let i = thread_rng().gen_range(0..bpe.num_tokens());
        let s = bpe.token_bytes(i as u32);
        text.extend_from_slice(s);
    }
    text
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use rand::{thread_rng, Rng};
    use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton};

    use bpe::appendable_encoder::AppendableEncoder;
    use bpe::byte_pair_encoding::BytePairEncoding;
    use bpe::interval_encoding::IntervalEncoding;
    use bpe::prependable_encoder::PrependableEncoder;
    use bpe_openai::{cl100k_base, o200k_base};

    use super::*;

    /// This test produces the output for the encoding example in the README.
    #[test]
    fn readme_example() {
        let tokens = ["a", "b", "c", "ab", "cb", "ac", "bb", "cbb", "acbb"];
        let bpe = BytePairEncoding::from_dictionary(tokens.map(|t| t.as_bytes().to_vec()), None);
        let text = "abacbb";
        let prefixes = (1..=text.len()).map(|end| &text[..end]).collect_vec();
        let all_prefix_tokens = prefixes
            .iter()
            .map(|prefix| {
                bpe.encode_via_backtracking(prefix.as_bytes())
                    .into_iter()
                    .map(|t| String::from_utf8(bpe.decode_tokens(&[t])).unwrap())
                    .collect_vec()
            })
            .collect_vec();
        let last_prefix_tokens = all_prefix_tokens
            .iter()
            .map(|tokens| tokens.last().unwrap())
            .collect_vec();

        println!("Token set: `{}`\n", tokens.join(" "));

        println!("All tokens for each prefix of `{text}`:\n");
        for (prefix, tokens) in prefixes.iter().zip(&all_prefix_tokens) {
            println!(
                "- `{prefix}` {}> `{}`",
                "-".repeat(text.len() + 2 - prefix.len()),
                tokens.join(" ")
            );
        }
        println!();

        println!("Last token for each prefix of `{text}`:\n");
        for (prefix, token) in prefixes.iter().zip(&last_prefix_tokens) {
            println!(
                "- `{prefix}` {}> `{token}`",
                "-".repeat(text.len() + 2 - prefix.len()),
            );
        }
        println!();

        println!("Encoding using last tokens of `{text}`:\n");
        let mut remaining = text.len();
        while remaining > 0 {
            let prefix = &text[..remaining];
            let token = last_prefix_tokens[remaining - 1];
            println!(
                "- `{prefix}` {}> `{token}`",
                "-".repeat(text.len() + 2 - prefix.len()),
            );
            remaining -= token.len();
        }
        println!("- `<empty>`");
    }

    #[test]
    fn test_appendable_encoder() {
        let bpe = &cl100k_base().bpe;
        let mut enc = AppendableEncoder::new(bpe);
        let input_string = create_test_bytes(bpe, 100);
        for (i, c) in input_string.iter().enumerate() {
            assert_eq!(enc.token_count(), bpe.count(&input_string[0..i]));
            enc.push(*c);
        }
    }

    #[test]
    fn test_correctness_cl100k() {
        // This is quite a challenging test case...
        let test_string = std::str::from_utf8(&[
            125, 34, 10, 10, 46, 109, 107, 100, 105, 114, 115, 32, 102, 100, 115, 32, 97, 100, 105,
            112, 105, 115, 105, 99, 105, 110, 103, 105, 116, 121, 69, 110, 103, 105, 110, 101, 32,
            69, 67, 105, 114, 105, 101, 32, 111, 112, 116, 105, 109, 97, 108, 95, 68, 65, 32, 111,
            102, 102, 101, 110, 100,
        ])
        .unwrap();
        let time = Instant::now();
        let bpe = &cl100k_base().bpe;
        println!("{:?}", time.elapsed());
        let encoded1 = cl100k_base_singleton()
            .lock()
            .encode_ordinary(test_string)
            .into_iter()
            .collect_vec();
        let encoded2 = bpe.encode_via_backtracking(test_string.as_bytes());
        assert_eq!(encoded1, encoded2);
        let encoded3 = bpe.encode_via_table(test_string.as_bytes());
        assert_eq!(encoded1, encoded3);
        let encoded4 = bpe.encode_via_bitfield(test_string.as_bytes());
        assert_eq!(encoded1, encoded4);
    }

    #[test]
    fn test_correctness_o200k() {
        // This is quite a challenging test case...
        let test_string = std::str::from_utf8(&[
            125, 34, 10, 10, 46, 109, 107, 100, 105, 114, 115, 32, 102, 100, 115, 32, 97, 100, 105,
            112, 105, 115, 105, 99, 105, 110, 103, 105, 116, 121, 69, 110, 103, 105, 110, 101, 32,
            69, 67, 105, 114, 105, 101, 32, 111, 112, 116, 105, 109, 97, 108, 95, 68, 65, 32, 111,
            102, 102, 101, 110, 100,
        ])
        .unwrap();
        let time = Instant::now();
        let bpe = &o200k_base().bpe;
        println!("{:?}", time.elapsed());
        let encoded1 = o200k_base_singleton()
            .lock()
            .encode_ordinary(test_string)
            .into_iter()
            .collect_vec();
        let encoded2 = bpe.encode_via_backtracking(test_string.as_bytes());
        assert_eq!(encoded1, encoded2);
        let encoded3 = bpe.encode_via_table(test_string.as_bytes());
        assert_eq!(encoded1, encoded3);
        let encoded4 = bpe.encode_via_bitfield(test_string.as_bytes());
        assert_eq!(encoded1, encoded4);
    }

    #[test]
    fn test_bpe_equivalence() {
        let bpe = &cl100k_base().bpe;
        for tokens in [10, 1000, 10000] {
            for _ in 0..5 {
                let test_input = create_test_bytes(bpe, tokens);
                let encoded1 = bpe.encode_via_backtracking(&test_input);
                let encoded2 = bpe.encode_via_bitfield(&test_input);
                assert_eq!(encoded1, encoded2, "{} {}", encoded1.len(), encoded2.len());
            }
        }
    }

    #[test]
    fn test_interval_count() {
        let bpe = &cl100k_base().bpe;
        let text = create_test_bytes(bpe, 10000);
        let intervals = IntervalEncoding::new(bpe, &text);
        for _ in 0..1000 {
            let start = thread_rng().gen_range(0..text.len());
            let end = thread_rng().gen_range(0..text.len());
            let range = start.min(end)..start.max(end);
            assert_eq!(
                intervals.count(range.clone()),
                bpe.encode_via_backtracking(&text[range]).len()
            );
        }
    }

    #[test]
    fn test_prependable_encoder() {
        let bpe = &cl100k_base().bpe;
        let mut enc = PrependableEncoder::new(bpe);
        let input_string = create_test_bytes(bpe, 100);
        for (i, c) in input_string.iter().enumerate().rev() {
            enc.push(*c);
            assert_eq!(enc.token_count(), bpe.count(&input_string[i..]));
        }
    }
}

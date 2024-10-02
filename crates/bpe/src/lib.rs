pub mod appendable_encoder;
pub mod backtrack_encoder;
mod bitfield;
pub mod byte_pair_encoding;
pub mod interval_encoding;
pub mod prependable_encoder;

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::byte_pair_encoding::BytePairEncoding;

    /// This test produces the output for the encoding example in the README.
    #[test]
    fn readme_example() {
        let tokens = ["a", "b", "c", "ab", "cb", "ac"].map(|t| t.as_bytes().to_vec());
        let bpe = BytePairEncoding::from_dictionary(tokens, None);
        let text = "abacb";
        let prefixes = (1..=text.len()).map(|end| &text[..end]).collect_vec();
        let all_prefix_tokens = prefixes
            .iter()
            .map(|prefix| {
                bpe.encode_via_backtracking(prefix.as_bytes())
                    .into_iter()
                    .map(|t| unsafe { String::from_utf8_unchecked(bpe.decode_tokens(&[t])) })
                    .collect_vec()
            })
            .collect_vec();
        let last_prefix_tokens = all_prefix_tokens
            .iter()
            .map(|tokens| tokens.last().unwrap())
            .collect_vec();

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

        println!("Tokenization of `{text}`:\n");
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
}

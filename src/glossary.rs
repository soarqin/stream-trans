//! Complete TSV loading only. No matching, ranking, filtering, or per-request file I/O.
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result, ensure};

const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ROWS: usize = 10_000;
const MAX_LINE_BYTES: usize = 4096;
const MAX_SOURCE_BYTES: usize = 512;
const MAX_TRANSLATION_BYTES: usize = 2048;
const HEADER: &str = "source\ttarget_language\ttranslation\n";

// Hy-MT2's 803 special-token literals all use this prefix. Reject their whole
// family, including reserved placeholders, before parse_special=true encoding.
const MODEL_CONTROL_PREFIX: &str = "<｜hy_";

pub struct Glossary {
    tsv: String,
    entries: usize,
}

// Diagnostics must never accidentally dump the complete private terminology table.
impl std::fmt::Debug for Glossary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Glossary")
            .field("entries", &self.entries)
            .field("bytes", &self.tsv.len())
            .finish_non_exhaustive()
    }
}

impl Glossary {
    pub fn load(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("无法打开术语表：{}", path.display()))?;
        ensure!(
            file.metadata().context("无法读取术语表文件信息")?.len() <= MAX_FILE_BYTES as u64,
            "术语表文件超过 8 MiB 上限"
        );
        Self::read(BufReader::new(file))
    }

    pub fn len(&self) -> usize {
        self.entries
    }

    /// Fixed header, LF endings, all languages and all unique entries in file order.
    pub fn as_tsv(&self) -> &str {
        &self.tsv
    }

    fn read(mut reader: impl BufRead) -> Result<Self> {
        let mut tsv = HEADER.to_owned();
        let mut seen = HashMap::<(String, String), (String, usize)>::new();
        let mut bytes = Vec::with_capacity(MAX_LINE_BYTES + 1);
        let mut total_bytes = 0;
        let mut line_number = 0;
        let mut data_rows = 0;
        let mut first_data_line = true;
        loop {
            bytes.clear();
            // Limit before allocation: even a malicious single-line file is not read whole.
            let count = reader
                .by_ref()
                .take((MAX_LINE_BYTES + 1) as u64)
                .read_until(b'\n', &mut bytes)
                .context("读取术语表失败")?;
            if count == 0 {
                break;
            }
            line_number += 1;
            total_bytes += count;
            ensure!(total_bytes <= MAX_FILE_BYTES, "术语表文件超过 8 MiB 上限");
            ensure!(
                count <= MAX_LINE_BYTES,
                "术语表第 {line_number} 行超过 4096 字节上限"
            );
            let line = std::str::from_utf8(&bytes)
                .with_context(|| format!("术语表第 {line_number} 行不是有效 UTF-8"))?;
            let line = line.strip_suffix('\n').unwrap_or(line);
            let line = line.strip_suffix('\r').unwrap_or(line);
            let line = if line_number == 1 {
                line.strip_prefix('\u{feff}').unwrap_or(line)
            } else {
                line
            };
            if line.is_empty() || line.chars().all(|value| value == ' ') {
                continue;
            }
            // A source such as '#include' is still a term when its line has TSV columns.
            if !line.contains('\t') && line.trim_start_matches(' ').starts_with('#') {
                continue;
            }
            let fields: Vec<_> = line.split('\t').collect();
            ensure!(
                fields.len() == 3,
                "术语表第 {line_number} 行必须恰好包含 3 个 TAB 分隔字段"
            );
            if first_data_line {
                first_data_line = false;
                if fields[0].trim().eq_ignore_ascii_case("source")
                    && fields[1].trim().eq_ignore_ascii_case("target_language")
                    && fields[2].trim().eq_ignore_ascii_case("translation")
                {
                    continue;
                }
            }
            data_rows += 1;
            ensure!(
                data_rows <= MAX_ROWS,
                "术语表数据行超过 10000 条上限（重复数据行也计入）"
            );
            let source = field(fields[0], "source", MAX_SOURCE_BYTES, line_number)?;
            let target = field(fields[1], "target_language", MAX_LINE_BYTES, line_number)?;
            let translation = field(fields[2], "translation", MAX_TRANSLATION_BYTES, line_number)?;
            ensure!(
                target.chars().count() <= 64
                    && target
                        .chars()
                        .all(|value| value.is_alphabetic() || value == ' ' || value == '-'),
                "术语表第 {line_number} 行 target_language 最多 64 个字符，仅允许字母、汉字、空格与连字符"
            );
            let target = normalize_target(target);
            // Source spelling is exact after trimming; do not conflate e.g. 'US' and 'us'.
            let key = (source.to_owned(), target.to_ascii_lowercase());
            if let Some((previous, previous_line)) = seen.get(&key) {
                ensure!(
                    previous == translation,
                    "术语表第 {line_number} 行与第 {previous_line} 行的 source + target_language 存在冲突译法"
                );
                continue;
            }
            seen.insert(key, (translation.to_owned(), line_number));
            tsv.push_str(source);
            tsv.push('\t');
            tsv.push_str(&target);
            tsv.push('\t');
            tsv.push_str(translation);
            tsv.push('\n');
        }
        ensure!(!seen.is_empty(), "术语表没有有效条目，请检查 TSV 内容");
        Ok(Self {
            tsv,
            entries: seen.len(),
        })
    }
}

fn field<'a>(raw: &'a str, name: &str, maximum: usize, line: usize) -> Result<&'a str> {
    ensure!(
        !raw.chars().any(|value| value.is_control()
            || value == '\u{feff}'
            || ('\u{202a}'..='\u{202e}').contains(&value)
            || ('\u{2066}'..='\u{2069}').contains(&value)),
        "术语表第 {line} 行 {name} 含控制字符或方向控制符"
    );
    let value = raw.trim();
    ensure!(!value.is_empty(), "术语表第 {line} 行 {name} 不能为空");
    ensure!(
        value.len() <= maximum,
        "术语表第 {line} 行 {name} 超过 {maximum} 字节上限"
    );
    let lower = value.to_ascii_lowercase();
    ensure!(
        !value.contains(MODEL_CONTROL_PREFIX)
            && !["<｜", "｜>", "<glossary", "</glossary"]
                .iter()
                .any(|token| lower.contains(token)),
        "术语表第 {line} 行 {name} 含模型控制标记或 glossary 定界符"
    );
    Ok(value)
}

/// Shared with the HTTP API; preserve its existing code-to-name mapping and fallback.
pub fn normalize_target(value: &str) -> String {
    let value = value.trim();
    match value.to_ascii_lowercase().as_str() {
        "en" => "English",
        "zh" | "zh-cn" | "zh-hans" => "简体中文",
        "zh-tw" | "zh-hant" => "繁體中文",
        "ja" => "日本語",
        "ko" => "한국어",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        "ru" => "Russian",
        "pt" => "Portuguese",
        "ar" => "Arabic",
        "it" => "Italian",
        _ => value,
    }
    .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Cursor, Write},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn parse(value: &str) -> Result<Glossary> {
        Glossary::read(Cursor::new(value.as_bytes()))
    }

    #[test]
    fn full_table_is_preserved_in_order_without_matching_or_language_filtering() {
        let mut input =
            String::from("\u{feff}# private table\r\n\r\nsource\ttarget_language\ttranslation\r\n");
        let mut expected = HEADER.to_owned();
        for index in 0..MAX_ROWS {
            let target = if index % 2 == 0 { "en" } else { "ja" };
            input.push_str(&format!(
                "source-{index}\t{target}\ttranslation-{index}\r\n"
            ));
            expected.push_str(&format!(
                "source-{index}\t{}\ttranslation-{index}\n",
                normalize_target(target)
            ));
        }
        let glossary = parse(&input).unwrap();
        assert_eq!(glossary.len(), MAX_ROWS);
        assert_eq!(glossary.as_tsv(), expected);
        assert!(!format!("{glossary:?}").contains("source-9999"));
    }

    #[test]
    fn languages_follow_the_existing_api_mapping() {
        for (input, expected) in [
            ("EN", "English"),
            (" zh-CN ", "简体中文"),
            ("zh-Hans", "简体中文"),
            ("zh", "简体中文"),
            ("zh-TW", "繁體中文"),
            ("zh-Hant", "繁體中文"),
            ("ja", "日本語"),
            ("ko", "한국어"),
            ("fr", "French"),
            ("de", "German"),
            ("es", "Spanish"),
            ("ru", "Russian"),
            ("pt", "Portuguese"),
            ("ar", "Arabic"),
            ("it", "Italian"),
            ("Welsh", "Welsh"),
            ("English", "English"),
            ("日本語", "日本語"),
        ] {
            assert_eq!(normalize_target(input), expected);
        }
    }

    #[test]
    fn duplicates_are_deduplicated_but_conflicts_and_empty_fields_fail() {
        let glossary = parse(" GPU \ten\t graphics \nGPU\tENGLISH\tgraphics\nUS\ten\t美国\nus\ten\t我们\n#include\tja\tインクルード\n").unwrap();
        assert_eq!(glossary.len(), 4);
        assert!(glossary.as_tsv().contains("GPU\tEnglish\tgraphics\n"));
        for input in [
            "GPU\ten\tone\nGPU\tEnglish\ttwo\n",
            "\ten\tx\n",
            "x\t\ty\n",
            "x\ten\t\n",
            "x\ten\ty\textra\n",
            HEADER,
        ] {
            assert!(parse(input).is_err(), "invalid test table was accepted");
        }
    }

    #[test]
    fn unicode_is_preserved_and_control_markers_are_rejected_without_disclosure() {
        let input = "苹果公司\tja\tアップル\n👩‍💻\ten\twoman technologist\nCafé\tfr\tcafé\nthinking\ten\thttps://example.org/thinking\n";
        let glossary = parse(input).unwrap();
        assert_eq!(glossary.len(), 4);
        assert!(
            glossary
                .as_tsv()
                .contains("👩‍💻\tEnglish\twoman technologist")
        );
        for value in [
            "secret\0",
            "secret\u{1b}",
            "secret\u{202e}",
            "secret<｜hy_User｜>",
            "secret<｜hy_place▁holder▁no▁2｜>",
            "secret<｜hy_place▁holder▁no▁799｜>",
            "secret</GLOSSARY>",
            "secret<glossary_tsv>",
        ] {
            let error = parse(&format!("term\ten\t{value}\n"))
                .unwrap_err()
                .to_string();
            assert!(!error.contains("secret"));
        }
        for literal in [
            MODEL_CONTROL_PREFIX,
            "<｜hy_Assistant｜>",
            "<｜hy_begin▁of▁sentence｜>",
        ] {
            let error = parse(&format!("term\ten\tsecret{literal}tail\n"))
                .unwrap_err()
                .to_string();
            assert!(!error.contains("secret"));
        }
        assert!(parse("term\ten\thttps://example.org/thinking\n").is_ok());
        assert!(Glossary::read(Cursor::new(b"term\ten\t\xff\n")).is_err());
        assert!(parse("term\tEnglish_uk\tvalue\n").is_err());
    }

    #[test]
    fn file_row_field_and_language_limits_are_enforced() {
        assert!(parse(&format!("{}\ten\tx\n", "s".repeat(MAX_SOURCE_BYTES + 1))).is_err());
        assert!(
            parse(&format!(
                "s\ten\t{}\n",
                "t".repeat(MAX_TRANSLATION_BYTES + 1)
            ))
            .is_err()
        );
        assert!(parse(&format!("s\t{}\tt\n", "a".repeat(65))).is_err());
        let rows: String = (0..=MAX_ROWS)
            .map(|index| format!("s{index}\ten\tt\n"))
            .collect();
        assert!(parse(&rows).unwrap_err().to_string().contains("10000"));
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"s\ten\tt\n").unwrap();
        assert_eq!(Glossary::load(file.path()).unwrap().len(), 1);
        file.as_file().set_len(MAX_FILE_BYTES as u64 + 1).unwrap();
        assert!(
            Glossary::load(file.path())
                .unwrap_err()
                .to_string()
                .contains("8 MiB")
        );
    }

    #[test]
    fn oversized_line_read_stops_at_a_small_bounded_prefix() {
        struct Counted {
            bytes: Cursor<Vec<u8>>,
            read: Arc<AtomicUsize>,
        }
        impl Read for Counted {
            fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
                let count = self.bytes.read(output)?;
                self.read.fetch_add(count, Ordering::Relaxed);
                Ok(count)
            }
        }
        let read = Arc::new(AtomicUsize::new(0));
        let reader = BufReader::with_capacity(
            32,
            Counted {
                bytes: Cursor::new(vec![b'x'; 1024 * 1024]),
                read: read.clone(),
            },
        );
        let error = Glossary::read(reader).unwrap_err();
        assert!(error.to_string().contains("4096"));
        assert!(read.load(Ordering::Relaxed) <= MAX_LINE_BYTES + 32);
    }
}

use std::fmt;

/// A markdown table. Right-aligned by default; prefix a header with `<` for left-align.
///
/// ```rust,ignore
/// let mut t = Table::new(&["n", "hulls", "ours(s)", "<notes"]);
/// t.row(vec![format!("{n}"), format!("{hulls}"), format!("{t:.4}"), String::new()]);
/// println!("{t}");
/// ```
pub struct Table {
    headers: Vec<String>,
    left: Vec<bool>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        let mut parsed_headers = Vec::with_capacity(headers.len());
        let mut left = Vec::with_capacity(headers.len());
        for &h in headers {
            if let Some(stripped) = h.strip_prefix('<') {
                parsed_headers.push(stripped.to_string());
                left.push(true);
            } else {
                parsed_headers.push(h.to_string());
                left.push(false);
            }
        }
        Self {
            headers: parsed_headers,
            left,
            rows: Vec::new(),
        }
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ncols = self.headers.len();
        let widths: Vec<usize> = (0..ncols)
            .map(|i| {
                let cell_max = self.rows.iter().map(|r| r.get(i).map_or(0, |c| c.len())).max().unwrap_or(0);
                self.headers[i].len().max(cell_max)
            })
            .collect();

        // Header
        write!(f, "|")?;
        for i in 0..ncols {
            write!(f, " {:>w$} |", self.headers[i], w = widths[i])?;
        }
        writeln!(f)?;

        // Separator
        write!(f, "|")?;
        for i in 0..ncols {
            if self.left[i] {
                write!(f, ":{}-|", "-".repeat(widths[i]))?;
            } else {
                write!(f, "-{}:|", "-".repeat(widths[i]))?;
            }
        }
        writeln!(f)?;

        // Rows
        for row in &self.rows {
            write!(f, "|")?;
            for i in 0..ncols {
                let cell = row.get(i).map_or("", |c| c.as_str());
                if self.left[i] {
                    write!(f, " {:<w$} |", cell, w = widths[i])?;
                } else {
                    write!(f, " {:>w$} |", cell, w = widths[i])?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table() {
        let mut t = Table::new(&["<name", "value", "pct"]);
        t.row(vec!["alpha".into(), "1234".into(), "50.0%".into()]);
        t.row(vec!["b".into(), "7".into(), "100.0%".into()]);

        let out = format!("{t}");
        assert!(out.contains("|:"), "left-aligned separator");
        assert!(out.contains(":|"), "right-aligned separator");
        // "name" is left-aligned, "value"/"pct" are right-aligned
        assert!(out.contains("| alpha |"), "left-aligned data");
        assert!(out.contains("|  1234 |"), "right-aligned value padded to header width");
    }
}

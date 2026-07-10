//! Schema Field Mapper — production Rust version (no LLM).
//!
//! PROJECT CONTEXT:
//! Interview-challenge pipeline: map every field of a legacy MySQL HR schema
//! (legacy_hrm) to its semantic equivalent in a MongoDB schema
//! (people_platform), emitting one JSON mapping document plus a metadata
//! report and a run log. This Rust version demonstrates that the task is a
//! classic schema-matching problem solvable with deterministic techniques
//! (abbreviation dictionaries, token-set similarity, Levenshtein distance,
//! type compatibility, and explicit domain rules) rather than an LLM.
//!
//! INPUTS (hardcoded paths, current working directory):
//!   sql_schema_legacy_hrm.txt      — pseudo-JSON MySQL schema with -- comments
//!   mongo_schema_people_platform.txt — pseudo-JSON MongoDB schema with -- comments
//!
//! OUTPUTS (timestamped filenames, current working directory):
//!   schema_mapping_output_<ts>.json
//!   schema_mapper_metadata_report_<ts>.json
//!   schema_mapper_log_<ts>.log
//!
//! EXIT BEHAVIOR: 0 on success, 1 on any handled failure (never panics in
//! a release build; all failures are detected, logged as a terse error
//! code, and the process exits cleanly).
//!
//! CASE-HANDLING FRAMEWORK: three-mode (test / debug / production-release).
//! Production-release error paths return only the 2-byte SmapError code.
//! Verbose diagnostics are gated behind #[cfg(debug_assertions)].

use std::cmp::Ordering;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

// =====================================================================
// Error type: single fieldless enum, u16 discriminants ARE the codes.
//
// Error-Code-Table (APPEND-ONLY; never renumber or reuse):
//   100-block: file I/O
//     101 ReadSqlSchemaFailed     — could not read SQL schema input file
//     102 ReadMongoSchemaFailed   — could not read Mongo schema input file
//     103 WriteMappingFailed      — could not write mapping output JSON
//     104 WriteMetadataFailed     — could not write metadata report JSON
//     105 WriteLogFailed          — could not write run log file
//   200-block: parsing
//     201 SqlParseNoFields        — SQL parse yielded zero fields
//     202 MongoParseNoFields      — Mongo parse yielded zero fields
//   300-block: matching
//     301 NoTablePairs            — no table/collection pair met threshold
//   400-block: environment
//     401 ClockBeforeEpoch        — system clock reports pre-1970 time
// =====================================================================

/// Project error type. Fieldless: 2 bytes, Copy, no heap, no payload
/// (payloads are the PII/leak vector this policy closes).
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u16)]
pub enum SmapError {
    ReadSqlSchemaFailed = 101,
    ReadMongoSchemaFailed = 102,
    WriteMappingFailed = 103,
    WriteMetadataFailed = 104,
    WriteLogFailed = 105,
    SqlParseNoFields = 201,
    MongoParseNoFields = 202,
    NoTablePairs = 301,
    ClockBeforeEpoch = 401,
}

impl SmapError {
    /// The terse numeric error-code. Available in ALL builds. No heap.
    pub fn code(self) -> u16 {
        self as u16
    }
}

// =====================================================================
// Timestamp (no chrono): civil-date conversion, Howard Hinnant algorithm.
// =====================================================================

/// Convert days-since-epoch to (year, month, day). Pure integer math,
/// panic-free for the full i64 day range used here (post-1970 only,
/// guarded by the ClockBeforeEpoch check at the call site).
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = (z - era * 146_097) as u64; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (year + if month <= 2 { 1 } else { 0 }, month, day)
}

/// Produce (iso_8601_utc, filename_safe) timestamp strings from the
/// system clock. Returns Err(ClockBeforeEpoch) if the clock is before
/// 1970 (hardware/RTC failure case — handled, not assumed impossible).
fn make_timestamps() -> Result<(String, String), SmapError> {
    let seconds_since_epoch = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_secs(),
        Err(_detail) => {
            #[cfg(debug_assertions)]
            eprintln!("SMAP-401: make_timestamps: clock before epoch: {}", _detail);
            return Err(SmapError::ClockBeforeEpoch);
        }
    };
    let days = (seconds_since_epoch / 86_400) as i64;
    let secs_of_day = seconds_since_epoch % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let iso = format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    );
    let file_safe = format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        year, month, day, hour, minute, second
    );
    Ok((iso, file_safe))
}

// =====================================================================
// Tokenization and abbreviation/synonym canonicalization.
// =====================================================================

/// Map a lowercase raw token to its canonical form using the domain
/// abbreviation/synonym dictionary. Unknown tokens pass through.
/// This dictionary encodes HR-schema domain knowledge (emp -> employee,
/// hire -> start, etc.) and is a documented, auditable matching input.
fn canonical_token(raw: &str) -> &str {
    match raw {
        "emp" | "employees" | "employee" => "employee",
        "dept" | "departments" | "department" => "department",
        "loc" | "locations" | "location" => "location",
        "cd" | "code" => "code",
        "dt" | "date" => "date",
        "nm" | "name" => "name",
        "sal" | "salary" => "salary",
        "mgr" | "manager" => "manager",
        "lvl" | "level" => "level",
        "stat" | "status" => "status",
        "ctr" | "center" => "center",
        "prov" | "province" => "province",
        "tz" | "timezone" => "timezone",
        "f" | "first" => "first",
        "l" | "last" => "last",
        "hire" => "start", // semantic synonym: hire date == start date
        "term" => "end",   // semantic synonym: termination date == end date
        other => other,
    }
}

/// Tokens carrying no matching signal in this domain; dropped before
/// comparison. Documented rationale: prefixes like "work_"/"office_" and
/// suffixes like "_ts" are container noise ("work_email" means "email").
fn is_noise_token(token: &str) -> bool {
    matches!(
        token,
        "work" | "office" | "rec" | "master" | "info" | "or" | "ts" | "at"
    )
}

/// Tokenize free-text comments: lowercase, split on non-alphanumeric,
/// dedupe. No abbreviation mapping, no noise removal — comments carry
/// standard references ("ISO 4217", "IANA timezone") verbatim.
fn comment_tokens(comment: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in comment.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if !tokens.contains(&current) {
                tokens.push(current.clone());
            }
            current.clear();
        }
    }
    if !current.is_empty() && !tokens.contains(&current) {
        tokens.push(current);
    }
    tokens
}

/// Split a field name into canonical lowercase tokens.
/// Handles snake_case, camelCase, leading underscores ("_id" -> ["id"]),
/// and drops noise tokens. Deterministic; no heap surprises.
fn tokenize(name: &str) -> Vec<String> {
    let mut raw_tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in name.chars() {
        if ch == '_' || ch == '.' || ch == '-' {
            if !current.is_empty() {
                raw_tokens.push(current.clone());
                current.clear();
            }
        } else if ch.is_ascii_uppercase() {
            if !current.is_empty() {
                raw_tokens.push(current.clone());
                current.clear();
            }
            current.push(ch.to_ascii_lowercase());
        } else if ch.is_ascii_alphanumeric() {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        raw_tokens.push(current);
    }
    let mut canonical: Vec<String> = Vec::new();
    for token in &raw_tokens {
        let mapped = canonical_token(token.as_str());
        if !is_noise_token(mapped) && !mapped.is_empty() {
            let owned = mapped.to_string();
            if !canonical.contains(&owned) {
                canonical.push(owned);
            }
        }
    }
    canonical
}

// =====================================================================
// Levenshtein distance (two-row iterative, bounded).
// =====================================================================

/// Maximum compared length; longer inputs are truncated. Bounds the loop
/// per the bounded-loop rule; schema identifiers are far shorter.
const LEVENSHTEIN_MAX_LEN: usize = 128;

/// Iterative two-row Levenshtein distance over ASCII-lowercased bytes.
/// No recursion, bounded loops, no panicking indexing (indices are
/// derived from the same lengths that size the buffers).
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_bytes: Vec<u8> = a.bytes().take(LEVENSHTEIN_MAX_LEN).collect();
    let b_bytes: Vec<u8> = b.bytes().take(LEVENSHTEIN_MAX_LEN).collect();
    if a_bytes.is_empty() {
        return b_bytes.len();
    }
    if b_bytes.is_empty() {
        return a_bytes.len();
    }
    let mut previous_row: Vec<usize> = (0..=b_bytes.len()).collect();
    let mut current_row: Vec<usize> = vec![0; b_bytes.len() + 1];
    for (i, a_byte) in a_bytes.iter().enumerate() {
        current_row[0] = i + 1;
        for (j, b_byte) in b_bytes.iter().enumerate() {
            let substitution_cost = if a_byte == b_byte { 0 } else { 1 };
            let deletion = previous_row[j + 1] + 1;
            let insertion = current_row[j] + 1;
            let substitution = previous_row[j] + substitution_cost;
            current_row[j + 1] = deletion.min(insertion).min(substitution);
        }
        std::mem::swap(&mut previous_row, &mut current_row);
    }
    previous_row[b_bytes.len()]
}

/// Normalized Levenshtein similarity in [0.0, 1.0].
fn levenshtein_similarity(a: &str, b: &str) -> f64 {
    let max_len = a.len().max(b.len());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein_distance(a, b);
    1.0 - (dist as f64 / max_len as f64)
}

// =====================================================================
// Input-file parser (line-oriented, tailored to the known pseudo-JSON
// format: `"name": TYPE ANNOTATIONS -- comment`, `"name": {`, `}`).
// =====================================================================

/// One parsed field from either input file.
/// For SQL: container = table name, path = column name.
/// For Mongo: container = collection name, path = dot path to the leaf
/// (e.g. "fullName.firstName").
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
#[derive(Clone)]
struct SchemaField {
    container: String,
    path: String,
    declared_type: String,
    annotations: String,
    comment: String,
}

/// Parse a schema file. `container_key` is "tables" (SQL) or
/// "collections" (Mongo). Unparseable lines are skipped (logged by the
/// caller via the returned skip count) — input validation, not a bug,
/// so this never panics in any mode.
fn parse_schema_text(text: &str, container_key: &str) -> (Vec<SchemaField>, usize) {
    let mut fields: Vec<SchemaField> = Vec::new();
    let mut skipped_lines: usize = 0;
    // Stack of open structures; "" = anonymous `{`.
    let mut stack: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Split off inline comment.
        let (code_part, comment_part) = match trimmed.split_once("--") {
            Some((c, m)) => (c.trim(), m.trim()),
            None => (trimmed, ""),
        };
        if code_part.is_empty() {
            continue;
        }
        if code_part.starts_with('}') {
            if stack.pop().is_none() {
                skipped_lines += 1; // unbalanced close: tolerate and continue
            }
            continue;
        }
        if code_part == "{" {
            stack.push(String::new());
            continue;
        }
        if !code_part.starts_with('"') {
            skipped_lines += 1;
            continue;
        }
        // Extract quoted name.
        let after_open_quote = &code_part[1..];
        let (name, remainder) = match after_open_quote.split_once('"') {
            Some((n, r)) => (n, r),
            None => {
                skipped_lines += 1;
                continue;
            }
        };
        let rest = match remainder.trim_start().strip_prefix(':') {
            Some(r) => r.trim(),
            None => {
                skipped_lines += 1;
                continue;
            }
        };
        if rest.starts_with('{') {
            stack.push(name.to_string());
            continue;
        }
        if rest.starts_with('"') {
            continue; // scalar metadata like "database": "legacy_hrm",
        }
        // Field line: first token is the declared type; remainder is
        // annotations (PRIMARY KEY, FK -> ..., UNIQUE, NOT NULL, etc.).
        let mut parts = rest.splitn(2, char::is_whitespace);
        let declared_type = match parts.next() {
            Some(t) => t.trim_end_matches(',').to_string(),
            None => {
                skipped_lines += 1;
                continue;
            }
        };
        let annotations = parts.next().unwrap_or("").trim().to_string();
        // Locate container_key in the stack; the field must sit under
        // container_key -> container_name [-> nested subdocs...].
        let key_position = stack.iter().position(|s| s == container_key);
        let named_after_key: Vec<&String> = match key_position {
            Some(pos) => stack[pos + 1..].iter().filter(|s| !s.is_empty()).collect(),
            None => {
                skipped_lines += 1;
                continue;
            }
        };
        if named_after_key.is_empty() {
            skipped_lines += 1;
            continue;
        }
        let container = named_after_key[0].clone();
        let mut path = String::new();
        for nested in &named_after_key[1..] {
            path.push_str(nested);
            path.push('.');
        }
        path.push_str(name);
        fields.push(SchemaField {
            container,
            path,
            declared_type,
            annotations,
            comment: comment_part.to_string(),
        });
    }
    (fields, skipped_lines)
}

// =====================================================================
// Type compatibility and transform description.
// =====================================================================

/// Coarse SQL base type extracted from a declared type like "VARCHAR(50)".
fn sql_base_type(declared: &str) -> String {
    let upper = declared.to_ascii_uppercase();
    match upper.split('(').next() {
        Some(base) => base.to_string(),
        None => upper,
    }
}

/// Extract the referenced table from a source annotation such as
/// "FK -> dept_info.dept_id" -> Some("dept_info"). None when no FK.
fn fk_target_table(annotations: &str) -> Option<String> {
    let after_marker = match annotations.split("FK ->").nth(1) {
        Some(rest) => rest.trim_start(),
        None => return None,
    };
    let target: String = after_marker
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if target.is_empty() {
        None
    } else {
        Some(target)
    }
}

/// Extract the referenced collection from a destination comment such as
/// "ref -> departments._id" -> Some("departments"). A "self-ref" comment
/// resolves to the field's own collection. None when no reference.
fn ref_target_collection(comment: &str, own_collection: &str) -> Option<String> {
    if comment.contains("self-ref") {
        return Some(own_collection.to_string());
    }
    let after_marker = match comment.split("ref ->").nth(1) {
        Some(rest) => rest.trim_start(),
        None => return None,
    };
    let target: String = after_marker
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if target.is_empty() {
        None
    } else {
        Some(target)
    }
}

/// Whether a SQL declared type is plausibly convertible to a BSON type.
/// Used only as a small scoring bonus, never as a hard gate.
fn types_compatible(sql_declared: &str, bson_type: &str) -> bool {
    let base = sql_base_type(sql_declared);
    match base.as_str() {
        "INT" | "BIGINT" | "SMALLINT" => {
            matches!(bson_type, "ObjectId" | "Number" | "Boolean")
        }
        "TINYINT" => matches!(bson_type, "Boolean" | "Number"),
        "VARCHAR" | "CHAR" | "TEXT" => matches!(bson_type, "String" | "Boolean"),
        "DATE" | "DATETIME" | "TIMESTAMP" => bson_type == "ISODate",
        "DECIMAL" | "FLOAT" | "DOUBLE" => bson_type == "Number",
        _ => false,
    }
}

/// Human-readable type-transform string. "code" marks a coded source
/// column; " enum" applies only to String destinations (a Boolean
/// destination is a plain Boolean, not an enum).
fn type_transform_text(source: &SchemaField, dest: &SchemaField, enum_coded: bool) -> String {
    let mut text = if enum_coded && dest.declared_type == "String" {
        format!(
            "{} code -> {} enum",
            source.declared_type, dest.declared_type
        )
    } else if enum_coded {
        format!("{} code -> {}", source.declared_type, dest.declared_type)
    } else {
        format!("{} -> {}", source.declared_type, dest.declared_type)
    };
    if dest.path.contains('.') {
        text.push_str(" (nested path)");
    }
    text
}

// =====================================================================
// Enum-code comment parsing (e.g. "A=Active, I=Inactive, T=Terminated").
// =====================================================================

/// Extract (code, meaning) pairs from a comment containing "X=Word" items.
/// Returns an empty vector when the comment carries no such pairs.
fn parse_enum_codes(comment: &str) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for segment in comment.split(',') {
        let piece = segment.trim();
        if let Some((code, meaning)) = piece.split_once('=') {
            let code_trimmed = code.trim();
            let meaning_trimmed = meaning.trim();
            // Accept only short single-token codes (defensive: comments may
            // contain '=' in other contexts).
            if !code_trimmed.is_empty()
                && code_trimmed.len() <= 3
                && !meaning_trimmed.is_empty()
                && code_trimmed.chars().all(|c| c.is_ascii_alphanumeric())
            {
                pairs.push((code_trimmed.to_string(), meaning_trimmed.to_string()));
            }
        }
    }
    pairs
}

// =====================================================================
// Matching engine.
// =====================================================================

/// Minimum score for a field mapping to be accepted.
const FIELD_MATCH_THRESHOLD: f64 = 0.55;

/// Minimum score for a table/collection pairing to be accepted.
const TABLE_MATCH_THRESHOLD: f64 = 0.50;

/// Weight applied to a token that also appears in the table/collection
/// names of the current pair (e.g. "department" inside dept_info ->
/// departments). Such tokens are context, weakly informative; a full
/// weight of 1.0 over-penalizes legacy prefix conventions like "dept_cd".
const CONTEXT_TOKEN_WEIGHT: f64 = 0.2;

/// Weighted Jaccard and overlap-coefficient similarity between two token
/// sets, where tokens found in `context_tokens` carry CONTEXT_TOKEN_WEIGHT
/// instead of 1.0. Returns (jaccard, overlap, intersection_count).
/// Guards all divisions (empty/all-context sets score 0.0).
fn weighted_token_scores(
    a: &[String],
    b: &[String],
    context_tokens: &[String],
) -> (f64, f64, usize) {
    if a.is_empty() || b.is_empty() {
        return (0.0, 0.0, 0);
    }
    let weight_of = |token: &String| -> f64 {
        if context_tokens.contains(token) {
            CONTEXT_TOKEN_WEIGHT
        } else {
            1.0
        }
    };
    let intersection_count = a.iter().filter(|t| b.contains(t)).count();
    let intersection_weight: f64 = a.iter().filter(|t| b.contains(t)).map(weight_of).sum();
    let a_weight: f64 = a.iter().map(weight_of).sum();
    let b_weight: f64 = b.iter().map(weight_of).sum();
    let union_weight = a_weight + b_weight - intersection_weight;
    let smaller_weight = a_weight.min(b_weight);
    if union_weight <= 0.0 || smaller_weight <= 0.0 {
        return (0.0, 0.0, 0);
    }
    (
        intersection_weight / union_weight,
        intersection_weight / smaller_weight,
        intersection_count,
    )
}

/// Which signal produced a field match. Recorded so the output
/// "reasoning" text states WHY a mapping exists (auditable pipeline).
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
#[derive(Copy, Clone, PartialEq, Eq)]
enum MatchMethod {
    /// Canonical token sets are identical (strongest signal).
    ExactTokens,
    /// Partial token overlap plus string-similarity support.
    TokenOverlap,
    /// Generalized domain rule: a two-valued coded column (enum pairs
    /// parsed from its comment) maps to a Boolean destination whose name
    /// tokens include one of the coded meanings.
    DomainRuleStatusToBoolean,
}

/// Leaf segment of a dot path ("employment.startDate" -> "startDate").
fn leaf_name(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}

/// Jaccard and overlap-coefficient similarity between two token sets.
/// Returns (jaccard, overlap, intersection_size). Guards division by
/// zero (empty sets score 0.0 — a should-not-happen defensive check,
/// since callers tokenize non-empty names, but names could be all-noise).
fn token_set_scores(a: &[String], b: &[String]) -> (f64, f64, usize) {
    if a.is_empty() || b.is_empty() {
        return (0.0, 0.0, 0);
    }
    let intersection = a.iter().filter(|t| b.contains(t)).count();
    let union = a.len() + b.len() - intersection;
    let smaller = a.len().min(b.len());
    if union == 0 || smaller == 0 {
        return (0.0, 0.0, 0);
    }
    (
        intersection as f64 / union as f64,
        intersection as f64 / smaller as f64,
        intersection,
    )
}

/// Bonus/penalty when both sides declare a reference and the targets
/// do / do not correspond under the confirmed table pairing.
const FK_GRAPH_BONUS: f64 = 0.15;

/// Score one source field against one destination field.
/// Signals: weighted token similarity (context tokens discounted),
/// Levenshtein on full canonical names, comment similarity, type
/// compatibility, primary-key->_id, reference-graph consistency, and a
/// generalized two-valued-enum -> Boolean domain rule.
fn score_field_pair(
    source: &SchemaField,
    dest: &SchemaField,
    context_tokens: &[String],
    table_pair_map: &[(String, String)],
) -> (f64, MatchMethod) {
    let source_tokens = tokenize(&source.path);
    let dest_tokens = tokenize(leaf_name(&dest.path));

    // --- Signal 1: weighted token-set similarity (context discounted) ---
    let (jaccard, overlap, intersection) =
        weighted_token_scores(&source_tokens, &dest_tokens, context_tokens);

    // --- Signal 2: Levenshtein on FULL canonical names (undiscounted,
    // so near-duplicate destinations like employeeCode vs department.code
    // remain distinguishable) ---
    let lev_sim = levenshtein_similarity(&source_tokens.concat(), &dest_tokens.concat());

    let mut score = 0.6 * jaccard + 0.3 * overlap + 0.1 * lev_sim;

    // --- Signal 3: comment similarity (standard references such as
    // "ISO 4217", "IANA timezone" appear verbatim on both sides) ---
    let source_comment_tokens = comment_tokens(&source.comment);
    let dest_comment_tokens = comment_tokens(&dest.comment);
    if !source_comment_tokens.is_empty() && !dest_comment_tokens.is_empty() {
        let shared = source_comment_tokens
            .iter()
            .filter(|t| dest_comment_tokens.contains(t))
            .count();
        let union = source_comment_tokens.len() + dest_comment_tokens.len() - shared;
        if union > 0 {
            score += 0.1 * (shared as f64 / union as f64);
        }
    }

    // --- Signal 4: type-compatibility bonus (never a hard gate) ---
    if types_compatible(&source.declared_type, &dest.declared_type) {
        score += 0.05;
    }

    // --- Signal 5: primary key maps to Mongo _id ---
    let source_is_primary_key = source
        .annotations
        .to_ascii_uppercase()
        .contains("PRIMARY KEY");
    if source_is_primary_key && leaf_name(&dest.path) == "_id" {
        score += 0.25;
    }

    // --- Signal 6: reference-graph consistency. When both sides declare
    // a reference target, corresponding targets (under the confirmed
    // table pairing) are strong evidence; mismatched targets are strong
    // counter-evidence (e.g. dept_head_id -> emp_master must not map to
    // the self-referencing parentDepartmentId). ---
    if let (Some(source_target), Some(dest_target)) = (
        fk_target_table(&source.annotations),
        ref_target_collection(&dest.comment, &dest.container),
    ) {
        let targets_correspond = table_pair_map
            .iter()
            .any(|(s, d)| *s == source_target && *d == dest_target);
        if targets_correspond {
            score += FK_GRAPH_BONUS;
        } else {
            score -= FK_GRAPH_BONUS;
        }
    }

    let token_method = if intersection > 0 && (jaccard - 1.0).abs() < f64::EPSILON {
        MatchMethod::ExactTokens
    } else {
        MatchMethod::TokenOverlap
    };
    let token_score = score.clamp(0.0, 0.99);

    // --- Signal 7: generalized domain rule (replaces the hardcoded
    // status->isActive rule). A source column whose comment defines
    // exactly two coded values (e.g. "A=Active, I=Inactive"), targeting
    // a Boolean destination whose name tokens include one of the coded
    // meanings ("active" in isActive), is a Boolean in disguise. ---
    let enum_pairs = parse_enum_codes(&source.comment);
    let boolean_enum_rule_applies = dest.declared_type == "Boolean"
        && enum_pairs.len() == 2
        && enum_pairs
            .iter()
            .any(|(_, meaning)| dest_tokens.iter().any(|t| t.eq_ignore_ascii_case(meaning)));
    if boolean_enum_rule_applies && token_score < 0.85 {
        return (0.85, MatchMethod::DomainRuleStatusToBoolean);
    }

    (token_score, token_method)
}

/// A confirmed table/collection pairing.
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
struct TablePair {
    source_table: String,
    destination_collection: String,
    score: f64,
}

/// Distinct containers in first-seen order (stable, deterministic output).
fn distinct_containers(fields: &[SchemaField]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for field in fields {
        if !names.contains(&field.container) {
            names.push(field.container.clone());
        }
    }
    names
}

/// Pair source tables with destination collections by name similarity
/// (token sets after abbreviation expansion + Levenshtein), greedy
/// one-to-one assignment above TABLE_MATCH_THRESHOLD.
fn pair_tables(sql_fields: &[SchemaField], mongo_fields: &[SchemaField]) -> Vec<TablePair> {
    let tables = distinct_containers(sql_fields);
    let collections = distinct_containers(mongo_fields);
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (ti, table) in tables.iter().enumerate() {
        for (ci, collection) in collections.iter().enumerate() {
            let table_tokens = tokenize(table);
            let collection_tokens = tokenize(collection);
            let (jaccard, overlap, _inter) = token_set_scores(&table_tokens, &collection_tokens);
            let lev = levenshtein_similarity(&table_tokens.concat(), &collection_tokens.concat());
            let score = (0.7 * jaccard + 0.2 * overlap + 0.1 * lev).min(0.99);
            if score >= TABLE_MATCH_THRESHOLD {
                candidates.push((score, ti, ci));
            }
        }
    }
    // Deterministic order: score descending, then names ascending.
    candidates.sort_by(
        |a, b| match b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal) {
            Ordering::Equal => (a.1, a.2).cmp(&(b.1, b.2)),
            other => other,
        },
    );
    let mut table_used = vec![false; tables.len()];
    let mut collection_used = vec![false; collections.len()];
    let mut pairs: Vec<TablePair> = Vec::new();
    for (score, ti, ci) in candidates {
        if !table_used[ti] && !collection_used[ci] {
            table_used[ti] = true;
            collection_used[ci] = true;
            pairs.push(TablePair {
                source_table: tables[ti].clone(),
                destination_collection: collections[ci].clone(),
                score,
            });
        }
    }
    // Preserve source-table order in output for readability.
    pairs.sort_by_key(|p| {
        tables
            .iter()
            .position(|t| *t == p.source_table)
            .unwrap_or(usize::MAX)
    });
    pairs
}

/// One accepted field mapping, ready for JSON rendering.
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
struct FieldMappingResult {
    source_field: String,
    destination_field: String,
    type_transform: String,
    confidence: f64,
    reasoning: String,
    notes: Option<String>,
}

/// Candidate-score record for the metadata report (top alternatives
/// considered per source field, for audit/review).
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
struct CandidateRecord {
    source_field: String,
    top_candidates: Vec<(String, f64)>,
}

/// Complete matching result for one table/collection pair.
#[cfg_attr(any(test, debug_assertions), derive(Debug))]
struct TablePairResult {
    source_table: String,
    destination_collection: String,
    confidence: f64,
    reasoning: String,
    mappings: Vec<FieldMappingResult>,
    unmapped_source_fields: Vec<String>,
    unmapped_destination_fields: Vec<String>,
    candidate_records: Vec<CandidateRecord>,
}

/// Build the reasoning sentence for one accepted mapping.
/// The sentence aggregates which pipeline signals caught the match,
/// per the project decision that "reasoning" is an auditable log of
/// why the mapping was included.
fn build_reasoning(
    source: &SchemaField,
    dest: &SchemaField,
    method: MatchMethod,
    score: f64,
) -> String {
    let source_tokens = tokenize(&source.path).join(", ");
    let dest_leaf = leaf_name(&dest.path);
    match method {
        MatchMethod::ExactTokens => format!(
            "Exact canonical-token match after abbreviation expansion \
             ('{}' and '{}' both reduce to [{}]); declared types {} and {}.",
            source.path, dest_leaf, source_tokens, source.declared_type, dest.declared_type
        ),
        MatchMethod::TokenOverlap => format!(
            "Token overlap after abbreviation expansion ('{}' -> [{}] vs '{}'), \
             supported by string similarity and type compatibility (score {:.2}).",
            source.path, source_tokens, dest_leaf, score
        ),
        MatchMethod::DomainRuleStatusToBoolean => format!(
            "Domain rule: two-valued coded column '{}' maps to Boolean '{}'.",
            source.path, dest.path
        ),
    }
}

/// Build the `notes` string for one accepted field mapping, or `None`
/// when no note applies.
///
/// PROJECT CONTEXT:
/// The assignment's output format requires a `notes` field on every
/// mapping: "any value-transform logic required, or null". This function
/// is the single place in the pipeline where note text is generated;
/// the JSON writer emits the returned string verbatim. If a note in the
/// output JSON is wrong, the fix is here and only here.
///
/// NOTE TYPES PRODUCED (in this order; several can apply to one mapping):
///
/// 1. PRIMARY-KEY MIGRATION NOTE — source column is a PRIMARY KEY and the
///    destination leaf is Mongo's `_id`. The legacy integer key cannot
///    become an ObjectId directly, so the migration needs an ID-generation
///    strategy, and the original key should be preserved for traceability.
///
/// 2. FOREIGN-KEY REMAPPING NOTE — source column is an FK and the
///    destination is an ObjectId reference. The legacy integer FK value
///    must be remapped to the new ObjectId of the referenced document.
///    Mutually exclusive with note 1 (a column is treated as PK first).
///
/// 3. BOOLEAN-INTEGER TRANSFORM — source is the MySQL boolean-integer
///    pattern TINYINT(1) and the destination is a native Boolean:
///    values 0/1 become false/true.
///
/// 4. ENUM-CODE TRANSFORM — the source column's comment defines coded
///    values (e.g. "A=Active, I=Inactive", extracted by
///    `parse_enum_codes`). The transform table depends on the
///    destination type:
///      - Boolean destination: each code becomes true/false, where the
///        meaning "Active" (case-insensitive) is true and anything else
///        is false (e.g. dept_stat: A -> true, I -> false).
///      - String destination: each code becomes its lowercased meaning
///        (e.g. rec_stat: A -> active, I -> inactive, T -> terminated).
///
/// FORMATTING POLICY (applies to every note, current and future):
/// - Each note is a complete English sentence and MUST end with a period.
///   This is a uniformity requirement across the output document; a
///   missing period is a defect (it has occurred once already).
/// - When multiple notes apply, they are joined with a single space into
///   one string, in the order listed above.
///
/// Returns `None` when no note type applies, which the JSON writer
/// renders as `null` per the assignment format.
fn build_notes(source: &SchemaField, dest: &SchemaField) -> Option<String> {
    let mut notes: Vec<String> = Vec::new();

    // Key-role facts about the source column, read from the parsed
    // SQL annotations (e.g. "PRIMARY KEY", "FK -> dept_info.dept_id").
    let source_is_primary_key = source
        .annotations
        .to_ascii_uppercase()
        .contains("PRIMARY KEY");
    let source_is_foreign_key = source.annotations.contains("FK");
    let dest_leaf = leaf_name(&dest.path);

    // --- Note type 1: primary key -> _id migration strategy ---
    // `else if` (not a second `if`): a PK column gets the PK note only,
    // never both PK and FK notes.
    if source_is_primary_key && dest_leaf == "_id" {
        notes.push(format!(
            "Store original {} as a legacy field for traceability; ID generation strategy required.",
            source.path
        ));
    // --- Note type 2: integer FK -> ObjectId reference remapping ---
    } else if source_is_foreign_key && dest.declared_type == "ObjectId" {
        notes.push(
            "ID remapping required: legacy integer foreign key becomes ObjectId reference."
                .to_string(),
        );
    }

    // --- Note type 3: TINYINT(1) boolean-integer -> native Boolean ---
    if source.declared_type.to_ascii_uppercase() == "TINYINT(1)" && dest.declared_type == "Boolean"
    {
        // Sentence policy: trailing period required (see doc comment).
        notes.push("Transform: 0 -> false, 1 -> true.".to_string());
    }

    // --- Note type 4: coded-enum column -> explicit value-transform table ---
    // Enum pairs come from the source column's inline comment
    // (e.g. "A=Active, I=Inactive, T=Terminated").
    let enum_pairs = parse_enum_codes(&source.comment);
    if !enum_pairs.is_empty() {
        if dest.declared_type == "Boolean" {
            // Boolean destination: "Active" (case-insensitive) -> true,
            // every other meaning -> false.
            let rendered: Vec<String> = enum_pairs
                .iter()
                .map(|(code, meaning)| {
                    let truth = meaning.eq_ignore_ascii_case("active");
                    format!("{} -> {}", code, truth)
                })
                .collect();
            // Sentence policy: trailing period required (see doc comment).
            notes.push(format!("Transform: {}.", rendered.join(", ")));
        } else if dest.declared_type == "String" {
            // String destination: code -> lowercased meaning, matching the
            // destination schema's lowercase enum convention
            // ("active / inactive / terminated").
            let rendered: Vec<String> = enum_pairs
                .iter()
                .map(|(code, meaning)| format!("{} -> {}", code, meaning.to_ascii_lowercase()))
                .collect();
            // Sentence policy: trailing period required (see doc comment).
            notes.push(format!("Transform: {}.", rendered.join(", ")));
        }
        // Other destination types: no enum note. The mapping itself is
        // unaffected; only the transform table is omitted.
    }

    // Join policy: multiple applicable notes become one space-separated
    // string; no notes at all becomes None (rendered as JSON null).
    if notes.is_empty() {
        None
    } else {
        Some(notes.join(" "))
    }
}

/// Margin (score minus runner-up score) at or above which the reported
/// confidence equals the raw score. Below it, confidence is reduced by
/// half the shortfall: a 0.71 winner over a 0.69 runner-up is genuinely
/// less certain than a 0.71 winner over a 0.30 runner-up.
const MARGIN_SAFE: f64 = 0.15;

/// Confidence from raw score and best-alternative score. Floor 0.50 so
/// an accepted mapping never reports below coin-flip confidence.
fn margin_adjusted_confidence(score: f64, runner_up_score: f64) -> f64 {
    let margin = score - runner_up_score;
    if margin >= MARGIN_SAFE {
        return score;
    }
    (score - 0.5 * (MARGIN_SAFE - margin)).max(0.50)
}

/// Round a score to two decimals for stable JSON output.
fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Match all fields within one confirmed table/collection pair.
/// Greedy one-to-one assignment: all (source, dest) scores are computed,
/// sorted descending, and assigned when both sides are free and the
/// score meets FIELD_MATCH_THRESHOLD. Deterministic tiebreaks.
fn match_fields_for_pair(
    pair: &TablePair,
    sql_fields: &[SchemaField],
    mongo_fields: &[SchemaField],
    table_pair_map: &[(String, String)],
) -> TablePairResult {
    let sources: Vec<&SchemaField> = sql_fields
        .iter()
        .filter(|f| f.container == pair.source_table)
        .collect();
    let dests: Vec<&SchemaField> = mongo_fields
        .iter()
        .filter(|f| f.container == pair.destination_collection)
        .collect();

    // Context tokens: table + collection name tokens, discounted during
    // field-token comparison (see weighted_token_scores).
    let mut context_tokens = tokenize(&pair.source_table);
    for token in tokenize(&pair.destination_collection) {
        if !context_tokens.contains(&token) {
            context_tokens.push(token);
        }
    }

    // Score every combination (bounded: |sources| x |dests|).
    let mut candidates: Vec<(f64, usize, usize, MatchMethod)> = Vec::new();
    for (si, source) in sources.iter().enumerate() {
        for (di, dest) in dests.iter().enumerate() {
            let (score, method) = score_field_pair(source, dest, &context_tokens, table_pair_map);
            if score > 0.0 {
                candidates.push((score, si, di, method));
            }
        }
    }
    candidates.sort_by(
        |a, b| match b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal) {
            Ordering::Equal => (a.1, a.2).cmp(&(b.1, b.2)),
            other => other,
        },
    );

    // Metadata: top-3 candidates per source field (audit trail).
    let mut candidate_records: Vec<CandidateRecord> = Vec::new();
    for (si, source) in sources.iter().enumerate() {
        let mut mine: Vec<(String, f64)> = candidates
            .iter()
            .filter(|(_, csi, _, _)| *csi == si)
            .map(|(score, _, di, _)| (dests[*di].path.clone(), round2(*score)))
            .collect();
        mine.truncate(3);
        candidate_records.push(CandidateRecord {
            source_field: source.path.clone(),
            top_candidates: mine,
        });
    }

    // Greedy assignment.
    let mut source_used = vec![false; sources.len()];
    let mut dest_used = vec![false; dests.len()];
    let mut assigned: Vec<(usize, usize, f64, MatchMethod, f64)> = Vec::new();
    for (score, si, di, method) in &candidates {
        if *score >= FIELD_MATCH_THRESHOLD && !source_used[*si] && !dest_used[*di] {
            // Runner-up: best alternative among destinations still FREE at
            // this moment. Destinations already claimed by stronger matches
            // are not real alternatives and must not depress confidence.
            let runner_up_score = candidates
                .iter()
                .filter(|(_, csi, cdi, _)| *csi == *si && *cdi != *di && !dest_used[*cdi])
                .map(|(s, _, _, _)| *s)
                .fold(0.0_f64, f64::max);
            source_used[*si] = true;
            dest_used[*di] = true;
            assigned.push((*si, *di, *score, *method, runner_up_score));
        }
    }
    assigned.sort_by_key(|(si, _, _, _, _)| *si);

    // Result building: destructure all FIVE tuple elements, including the
    // runner-up score stored during greedy assignment.
    let mut mappings: Vec<FieldMappingResult> = Vec::new();
    for (si, di, score, method, runner_up_score) in &assigned {
        let source = sources[*si];
        let dest = dests[*di];
        let enum_coded = !parse_enum_codes(&source.comment).is_empty()
            && (dest.declared_type == "String" || dest.declared_type == "Boolean");
        let confidence_value = margin_adjusted_confidence(*score, *runner_up_score);

        // When the margin penalty reduced confidence below the raw score,
        // say so in the reasoning, so the score/confidence gap in the
        // output JSON is explained rather than looking like an error.
        let mut reasoning = build_reasoning(source, dest, *method, *score);
        if round2(confidence_value) + 1e-9 < round2(*score) {
            reasoning
                .push_str(" Confidence reduced below score due to a close runner-up candidate.");
        }

        mappings.push(FieldMappingResult {
            source_field: source.path.clone(),
            destination_field: dest.path.clone(),
            type_transform: type_transform_text(source, dest, enum_coded),
            confidence: round2(confidence_value),
            reasoning,
            notes: build_notes(source, dest),
        });
    }

    let unmapped_source_fields: Vec<String> = sources
        .iter()
        .enumerate()
        .filter(|(si, _)| !source_used[*si])
        .map(|(_, f)| f.path.clone())
        .collect();
    let unmapped_destination_fields: Vec<String> = dests
        .iter()
        .enumerate()
        .filter(|(di, _)| !dest_used[*di])
        .map(|(_, f)| f.path.clone())
        .collect();

    TablePairResult {
        source_table: pair.source_table.clone(),
        destination_collection: pair.destination_collection.clone(),
        confidence: round2(pair.score),
        reasoning: format!(
            "Table '{}' and collection '{}' share canonical name tokens after abbreviation expansion.",
            pair.source_table, pair.destination_collection
        ),
        mappings,
        unmapped_source_fields,
        unmapped_destination_fields,
        candidate_records,
    }
}

/// Full pipeline core: pair tables, then match fields per pair.
/// Returns Err(NoTablePairs) when no pairing met threshold (defensive:
/// with the known inputs this indicates corrupt/wrong input files).
fn build_all_mappings(
    sql_fields: &[SchemaField],
    mongo_fields: &[SchemaField],
) -> Result<Vec<TablePairResult>, SmapError> {
    let pairs = pair_tables(sql_fields, mongo_fields);
    if pairs.is_empty() {
        #[cfg(debug_assertions)]
        eprintln!("SMAP-301: build_all_mappings: no table pairs met threshold");
        return Err(SmapError::NoTablePairs);
    }
    let pair_map: Vec<(String, String)> = pairs
        .iter()
        .map(|p| (p.source_table.clone(), p.destination_collection.clone()))
        .collect();
    let mut results: Vec<TablePairResult> = Vec::new();
    for pair in &pairs {
        results.push(match_fields_for_pair(
            pair,
            sql_fields,
            mongo_fields,
            &pair_map,
        ));
    }
    Ok(results)
}

// =====================================================================
// JSON rendering (no serde; manual escaping and indentation).
// =====================================================================

/// Escape a string for JSON string-literal context.
fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Render a JSON string value (quoted, escaped).
fn json_str(value: &str) -> String {
    format!("\"{}\"", json_escape(value))
}

/// Render a JSON array of strings on one line.
fn json_string_array(values: &[String]) -> String {
    let items: Vec<String> = values.iter().map(|v| json_str(v)).collect();
    format!("[{}]", items.join(", "))
}

/// Render the mapping output document (the assignment's required format).
fn render_mapping_json(results: &[TablePairResult], generated_at_iso: &str) -> String {
    let mut out = String::with_capacity(16 * 1024);
    out.push_str("{\n");
    out.push_str("  \"mapping_version\": \"1.0\",\n");
    out.push_str("  \"source\": \"legacy_hrm (MySQL)\",\n");
    out.push_str("  \"destination\": \"people_platform (MongoDB)\",\n");
    out.push_str(&format!(
        "  \"generated_at\": {},\n",
        json_str(generated_at_iso)
    ));
    out.push_str("  \"tables\": [\n");
    for (ti, table) in results.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"source_table\": {},\n",
            json_str(&table.source_table)
        ));
        out.push_str(&format!(
            "      \"destination_collection\": {},\n",
            json_str(&table.destination_collection)
        ));
        out.push_str(&format!("      \"confidence\": {:.2},\n", table.confidence));
        out.push_str(&format!(
            "      \"reasoning\": {},\n",
            json_str(&table.reasoning)
        ));
        out.push_str("      \"field_mappings\": [\n");
        for (fi, mapping) in table.mappings.iter().enumerate() {
            out.push_str("        {\n");
            out.push_str(&format!(
                "          \"source_field\": {},\n",
                json_str(&mapping.source_field)
            ));
            out.push_str(&format!(
                "          \"destination_field\": {},\n",
                json_str(&mapping.destination_field)
            ));
            out.push_str(&format!(
                "          \"type_transform\": {},\n",
                json_str(&mapping.type_transform)
            ));
            out.push_str(&format!(
                "          \"confidence\": {:.2},\n",
                mapping.confidence
            ));
            out.push_str(&format!(
                "          \"reasoning\": {},\n",
                json_str(&mapping.reasoning)
            ));
            match &mapping.notes {
                Some(text) => out.push_str(&format!("          \"notes\": {}\n", json_str(text))),
                None => out.push_str("          \"notes\": null\n"),
            }
            out.push_str(if fi + 1 < table.mappings.len() {
                "        },\n"
            } else {
                "        }\n"
            });
        }
        out.push_str("      ],\n");
        out.push_str(&format!(
            "      \"unmapped_source_fields\": {},\n",
            json_string_array(&table.unmapped_source_fields)
        ));
        out.push_str(&format!(
            "      \"unmapped_destination_fields\": {}\n",
            json_string_array(&table.unmapped_destination_fields)
        ));
        out.push_str(if ti + 1 < results.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

/// Render the metadata report (thresholds, parse stats, candidate audit).
fn render_metadata_json(
    results: &[TablePairResult],
    generated_at_iso: &str,
    sql_field_count: usize,
    sql_skipped: usize,
    mongo_field_count: usize,
    mongo_skipped: usize,
) -> String {
    let mut out = String::with_capacity(16 * 1024);
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"generated_at\": {},\n",
        json_str(generated_at_iso)
    ));
    out.push_str(&format!(
        "  \"field_match_threshold\": {:.2},\n",
        FIELD_MATCH_THRESHOLD
    ));
    out.push_str(&format!(
        "  \"table_match_threshold\": {:.2},\n",
        TABLE_MATCH_THRESHOLD
    ));
    out.push_str(&format!("  \"sql_fields_parsed\": {},\n", sql_field_count));
    out.push_str(&format!("  \"sql_lines_skipped\": {},\n", sql_skipped));
    out.push_str(&format!(
        "  \"mongo_fields_parsed\": {},\n",
        mongo_field_count
    ));
    out.push_str(&format!("  \"mongo_lines_skipped\": {},\n", mongo_skipped));
    out.push_str("  \"table_pairs\": [\n");
    for (ti, table) in results.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"source_table\": {},\n",
            json_str(&table.source_table)
        ));
        out.push_str(&format!(
            "      \"destination_collection\": {},\n",
            json_str(&table.destination_collection)
        ));
        out.push_str(&format!("      \"confidence\": {:.2},\n", table.confidence));
        out.push_str("      \"candidate_audit\": [\n");
        for (ci, record) in table.candidate_records.iter().enumerate() {
            let rendered: Vec<String> = record
                .top_candidates
                .iter()
                .map(|(path, score)| {
                    format!(
                        "{{\"destination\": {}, \"score\": {:.2}}}",
                        json_str(path),
                        score
                    )
                })
                .collect();
            out.push_str(&format!(
                "        {{\"source_field\": {}, \"top_candidates\": [{}]}}{}\n",
                json_str(&record.source_field),
                rendered.join(", "),
                if ci + 1 < table.candidate_records.len() {
                    ","
                } else {
                    ""
                }
            ));
        }
        out.push_str("      ]\n");
        out.push_str(if ti + 1 < results.len() {
            "    },\n"
        } else {
            "    }\n"
        });
    }
    out.push_str("  ]\n}\n");
    out
}

// =====================================================================
// Main pipeline (one-shot CLI; exit 0 on success, 1 on handled failure).
// =====================================================================

/// Hardcoded input paths per current project scope.
const SQL_SCHEMA_FILE_PATH: &str = "sql_schema_legacy_hrm.txt";
const MONGO_SCHEMA_FILE_PATH: &str = "mongo_schema_people_platform.txt";

/// Run the pipeline. Every fallible step is matched explicitly; on any
/// failure the terse error code is printed to stderr and 1 is returned.
/// The run log is written on both success and failure paths (best effort;
/// a log-write failure itself is reported by code but cannot recurse).
fn run() -> i32 {
    let mut log_lines: Vec<String> = Vec::new();

    // --- Step 1: timestamps ---
    let (generated_at_iso, file_safe_ts) = match make_timestamps() {
        Ok(pair) => pair,
        Err(error) => {
            eprintln!("SMAP error code: {}", error.code());
            return 1;
        }
    };
    let mapping_path = format!("schema_mapping_output_{}.json", file_safe_ts);
    let metadata_path = format!("schema_mapper_metadata_report_{}.json", file_safe_ts);
    let log_path = format!("schema_mapper_log_{}.log", file_safe_ts);
    log_lines.push(format!("run started: {}", generated_at_iso));

    // Inner closure-free pipeline with a single exit that writes the log.
    let outcome: Result<(), SmapError> = (|| {
        // --- Step 2: read inputs ---
        let sql_text = match fs::read_to_string(SQL_SCHEMA_FILE_PATH) {
            Ok(text) => text,
            Err(_detail) => {
                #[cfg(debug_assertions)]
                eprintln!("SMAP-101: run: read sql schema: {}", _detail);
                log_lines.push("error 101: could not read SQL schema input".to_string());
                return Err(SmapError::ReadSqlSchemaFailed);
            }
        };
        let mongo_text = match fs::read_to_string(MONGO_SCHEMA_FILE_PATH) {
            Ok(text) => text,
            Err(_detail) => {
                #[cfg(debug_assertions)]
                eprintln!("SMAP-102: run: read mongo schema: {}", _detail);
                log_lines.push("error 102: could not read Mongo schema input".to_string());
                return Err(SmapError::ReadMongoSchemaFailed);
            }
        };

        // --- Step 3: parse ---
        let (sql_fields, sql_skipped) = parse_schema_text(&sql_text, "tables");
        let (mongo_fields, mongo_skipped) = parse_schema_text(&mongo_text, "collections");
        log_lines.push(format!(
            "parsed sql: {} fields, {} lines skipped",
            sql_fields.len(),
            sql_skipped
        ));
        log_lines.push(format!(
            "parsed mongo: {} fields, {} lines skipped",
            mongo_fields.len(),
            mongo_skipped
        ));
        if sql_fields.is_empty() {
            log_lines.push("error 201: sql parse yielded zero fields".to_string());
            return Err(SmapError::SqlParseNoFields);
        }
        if mongo_fields.is_empty() {
            log_lines.push("error 202: mongo parse yielded zero fields".to_string());
            return Err(SmapError::MongoParseNoFields);
        }

        // --- Step 4: match ---
        let results = match build_all_mappings(&sql_fields, &mongo_fields) {
            Ok(r) => r,
            Err(error) => {
                log_lines.push(format!("error {}: matching failed", error.code()));
                return Err(error);
            }
        };
        for table in &results {
            log_lines.push(format!(
                "pair '{}' -> '{}' (score {:.2}): {} mapped, {} source unmapped, {} destination unmapped",
                table.source_table,
                table.destination_collection,
                table.confidence,
                table.mappings.len(),
                table.unmapped_source_fields.len(),
                table.unmapped_destination_fields.len()
            ));
            for mapping in &table.mappings {
                log_lines.push(format!(
                    "  {} -> {} ({:.2})",
                    mapping.source_field, mapping.destination_field, mapping.confidence
                ));
            }
        }

        // --- Step 5: write outputs ---
        let mapping_json = render_mapping_json(&results, &generated_at_iso);
        if let Err(_detail) = fs::write(&mapping_path, mapping_json) {
            #[cfg(debug_assertions)]
            eprintln!("SMAP-103: run: write mapping: {}", _detail);
            log_lines.push("error 103: could not write mapping output".to_string());
            return Err(SmapError::WriteMappingFailed);
        }
        log_lines.push(format!("wrote mapping: {}", mapping_path));

        let metadata_json = render_metadata_json(
            &results,
            &generated_at_iso,
            sql_fields.len(),
            sql_skipped,
            mongo_fields.len(),
            mongo_skipped,
        );
        if let Err(_detail) = fs::write(&metadata_path, metadata_json) {
            #[cfg(debug_assertions)]
            eprintln!("SMAP-104: run: write metadata: {}", _detail);
            log_lines.push("error 104: could not write metadata report".to_string());
            return Err(SmapError::WriteMetadataFailed);
        }
        log_lines.push(format!("wrote metadata report: {}", metadata_path));
        Ok(())
    })();

    // --- Step 6: write the run log (both success and failure paths) ---
    log_lines.push(match &outcome {
        Ok(()) => "run completed: success".to_string(),
        Err(error) => format!("run completed: failure, error code {}", error.code()),
    });
    let log_text = log_lines.join("\n") + "\n";
    if let Err(_detail) = fs::write(&log_path, log_text) {
        #[cfg(debug_assertions)]
        eprintln!("SMAP-105: run: write log: {}", _detail);
        eprintln!("SMAP error code: {}", SmapError::WriteLogFailed.code());
        return 1;
    }

    match outcome {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("SMAP error code: {}", error.code());
            1
        }
    }
}

fn main() {
    // One-shot CLI: clean exit with 0/1 per project scope. Never panics
    // in release builds; all failures are handled above.
    std::process::exit(run());
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod main_tests {
    use super::*;

    /// Embedded copy of the source schema (assignment Dataset A).
    const SQL_SCHEMA_TEXT: &str = r#"
{
  "database": "legacy_hrm",
  "type": "MySQL (Relational)",
  "tables": {
    "emp_master": {
      "emp_id":        INT            PRIMARY KEY
      "emp_cd":        VARCHAR(20)    UNIQUE NOT NULL    -- human-readable employee code
      "f_name":        VARCHAR(50)    NOT NULL
      "l_name":        VARCHAR(50)    NOT NULL
      "dob":           DATE
      "hire_dt":       DATETIME
      "term_dt":       DATETIME                         -- null if still active
      "dept_id":       INT            FK -> dept_info.dept_id
      "mgr_emp_id":    INT            FK -> emp_master.emp_id
      "job_lvl_cd":    VARCHAR(10)                      -- e.g. L1, L2, IC3, M1
      "base_sal":      DECIMAL(12,2)
      "sal_currency":  CHAR(3)                          -- ISO 4217, e.g. USD
      "work_email":    VARCHAR(120)   UNIQUE
      "work_phone":    VARCHAR(20)
      "office_loc_id": INT            FK -> locations.loc_id
      "is_remote":     TINYINT(1)                       -- 0 or 1
      "rec_stat":      CHAR(1)                          -- A=Active, I=Inactive, T=Terminated
      "created_ts":    DATETIME                         -- record creation timestamp
      "updated_ts":    DATETIME                         -- last update timestamp
    },
    "dept_info": {
      "dept_id":         INT            PRIMARY KEY
      "dept_cd":         VARCHAR(20)    UNIQUE
      "dept_nm":         VARCHAR(100)
      "parent_dept_id":  INT            FK -> dept_info.dept_id   -- self-referencing
      "dept_head_id":    INT            FK -> emp_master.emp_id
      "cost_ctr_cd":     VARCHAR(20)                    -- finance cost center code
      "dept_stat":       CHAR(1)                        -- A=Active, I=Inactive
    },
    "locations": {
      "loc_id":       INT            PRIMARY KEY
      "loc_cd":       VARCHAR(20)    UNIQUE
      "loc_nm":       VARCHAR(100)
      "city":         VARCHAR(80)
      "state_prov":   VARCHAR(80)
      "country_cd":   CHAR(2)                           -- ISO 3166-1 alpha-2
      "postal_cd":    VARCHAR(20)
      "tz_cd":        VARCHAR(50)                       -- IANA timezone
    }
  }
}
"#;

    /// Embedded copy of the target schema (assignment Dataset B).
    const MONGO_SCHEMA_TEXT: &str = r#"
{
  "database": "people_platform",
  "type": "MongoDB (Document)",
  "collections": {
    "employees": {
      "_id":                    ObjectId
      "employeeCode":           String           -- unique human-readable ID
      "fullName": {
        "firstName":            String
        "lastName":             String
      },
      "employment": {
        "startDate":            ISODate
        "endDate":              ISODate          -- null if currently employed
        "status":               String           -- active / inactive / terminated
        "jobLevel":             String           -- e.g. L1, IC3, M1
        "isRemote":             Boolean
        "managerId":            ObjectId         -- ref -> employees._id
      },
      "compensation": {
        "baseSalary":           Number
        "currency":             String           -- ISO 4217
      },
      "contact": {
        "email":                String
        "phone":                String
      },
      "department": {
        "departmentId":         ObjectId         -- ref -> departments._id
        "code":                 String
        "name":                 String
      },
      "location": {
        "locationId":           ObjectId         -- ref -> locations._id
        "code":                 String
        "name":                 String
        "city":                 String
        "country":              String           -- ISO 3166-1 alpha-2
        "timezone":             String           -- IANA timezone
      },
      "meta": {
        "createdAt":            ISODate
        "updatedAt":            ISODate
      }
    },
    "departments": {
      "_id":                    ObjectId
      "code":                   String
      "name":                   String
      "parentDepartmentId":     ObjectId         -- self-ref
      "headEmployeeId":         ObjectId         -- ref -> employees._id
      "costCenterCode":         String
      "isActive":               Boolean
    },
    "locations": {
      "_id":                    ObjectId
      "code":                   String
      "name":                   String
      "city":                   String
      "stateOrProvince":        String
      "country":                String           -- ISO 3166-1 alpha-2
      "postalCode":             String
      "timezone":               String
    }
  }
}
"#;

    // --- error path: no table pairs (code 301) ---

    #[test]
    fn smap_build_all_mappings_disjoint_schemas_with_code_301() {
        let (sql_fields, _s) = parse_schema_text(
            "{\n\"tables\": {\n\"zzz_qqq\": {\n\"aaa\": INT\n}\n}\n}",
            "tables",
        );
        let (mongo_fields, _m) = parse_schema_text(
            "{\n\"collections\": {\n\"www_rrr\": {\n\"bbb\": String\n}\n}\n}",
            "collections",
        );
        let err = build_all_mappings(&sql_fields, &mongo_fields).unwrap_err();
        assert_eq!(err, SmapError::NoTablePairs);
        assert_eq!(err.code(), 301);
    }

    // --- change 1: context discounting lifts legacy-prefix matches ---

    #[test]
    fn smap_context_discounting_lifts_dept_cd_confidence() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let dept = results
            .iter()
            .find(|t| t.source_table == "dept_info")
            .expect("dept_info pair must exist");
        let dept_cd = dept
            .mappings
            .iter()
            .find(|m| m.source_field == "dept_cd")
            .expect("dept_cd mapping must exist");
        assert_eq!(dept_cd.destination_field, "code");
        assert!(
            dept_cd.confidence >= 0.80,
            "was 0.68 pre-change; got {}",
            dept_cd.confidence
        );
    }

    // --- change 3: FK graph resolves the former 0.59/0.59 tie ---

    #[test]
    fn smap_fk_graph_separates_dept_head_id_from_parent() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let dept = results
            .iter()
            .find(|t| t.source_table == "dept_info")
            .expect("dept_info pair must exist");
        let head = dept
            .mappings
            .iter()
            .find(|m| m.source_field == "dept_head_id")
            .expect("dept_head_id mapping must exist");
        assert_eq!(head.destination_field, "headEmployeeId");
        assert!(
            head.confidence >= 0.80,
            "was tied 0.59 pre-change; got {}",
            head.confidence
        );
    }

    // --- change 2: comment similarity widens country_cd margin ---

    #[test]
    fn smap_comment_similarity_boosts_country_cd() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let locations = results
            .iter()
            .find(|t| t.source_table == "locations")
            .expect("locations pair must exist");
        let country = locations
            .mappings
            .iter()
            .find(|m| m.source_field == "country_cd")
            .expect("country_cd mapping must exist");
        assert_eq!(country.destination_field, "country");
        assert!(
            country.confidence >= 0.75,
            "was 0.71 pre-change; got {}",
            country.confidence
        );
    }

    // --- change 4: generalized enum->Boolean rule still fires ---

    #[test]
    fn smap_generalized_enum_boolean_rule_maps_dept_stat() {
        assert_eq!(
            mapped_destination("dept_info", "dept_stat"),
            Some("isActive".to_string())
        );
    }

    // --- change 5: margin-adjusted confidence, direct unit tests ---

    #[test]
    fn smap_margin_confidence_penalizes_thin_margins_only() {
        // Wide margin: confidence equals score.
        assert!((margin_adjusted_confidence(0.90, 0.50) - 0.90).abs() < 1e-9);
        // Thin margin: penalized below score, floored at 0.50.
        let thin = margin_adjusted_confidence(0.90, 0.88);
        assert!(thin < 0.90);
        assert!(thin >= 0.50);
        // Floor holds even for very thin low scores.
        assert!(margin_adjusted_confidence(0.56, 0.55) >= 0.50);
    }

    // --- tokenizer ---

    #[test]
    fn smap_tokenize_snake_case_with_abbreviations() {
        assert_eq!(tokenize("f_name"), vec!["first", "name"]);
        assert_eq!(tokenize("mgr_emp_id"), vec!["manager", "employee", "id"]);
        assert_eq!(tokenize("hire_dt"), vec!["start", "date"]);
    }

    #[test]
    fn smap_tokenize_camel_case_and_underscore_id() {
        assert_eq!(tokenize("firstName"), vec!["first", "name"]);
        assert_eq!(tokenize("_id"), vec!["id"]);
        assert_eq!(tokenize("stateOrProvince"), vec!["state", "province"]);
    }

    #[test]
    fn smap_tokenize_drops_noise_tokens() {
        assert_eq!(tokenize("work_email"), vec!["email"]);
        assert_eq!(tokenize("created_ts"), vec!["created"]);
        assert_eq!(tokenize("createdAt"), vec!["created"]);
        assert_eq!(tokenize("rec_stat"), vec!["status"]);
    }

    // --- levenshtein ---

    #[test]
    fn smap_levenshtein_known_distances() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
    }

    // --- enum-code comment parsing ---

    #[test]
    fn smap_parse_enum_codes_from_comment() {
        let pairs = parse_enum_codes("A=Active, I=Inactive, T=Terminated");
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0], ("A".to_string(), "Active".to_string()));
        assert_eq!(pairs[2], ("T".to_string(), "Terminated".to_string()));
    }

    #[test]
    fn smap_parse_enum_codes_ignores_plain_comment() {
        assert!(parse_enum_codes("record creation timestamp").is_empty());
    }

    // --- parser ---

    #[test]
    fn smap_parse_sql_schema_field_counts() {
        let (fields, _skipped) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        assert_eq!(fields.len(), 34); // 19 + 7 + 8
        let emp_count = fields
            .iter()
            .filter(|f| f.container == "emp_master")
            .count();
        assert_eq!(emp_count, 19);
    }

    #[test]
    fn smap_parse_mongo_schema_nested_paths() {
        let (fields, _skipped) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        assert_eq!(fields.len(), 40); // 25 + 7 + 8
        assert!(
            fields
                .iter()
                .any(|f| f.container == "employees" && f.path == "fullName.firstName")
        );
        assert!(
            fields
                .iter()
                .any(|f| f.container == "employees" && f.path == "employment.managerId")
        );
    }

    #[test]
    fn smap_parse_extracts_annotations_and_comments() {
        let (fields, _skipped) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let emp_id = fields
            .iter()
            .find(|f| f.path == "emp_id")
            .expect("emp_id must parse");
        assert!(emp_id.annotations.contains("PRIMARY KEY"));
        let rec_stat = fields
            .iter()
            .find(|f| f.path == "rec_stat")
            .expect("rec_stat must parse");
        assert!(rec_stat.comment.contains("A=Active"));
    }
    // --- table pairing ---

    #[test]
    fn smap_pair_tables_matches_all_three() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let pairs = pair_tables(&sql_fields, &mongo_fields);
        assert_eq!(pairs.len(), 3);
        let find = |table: &str| {
            pairs
                .iter()
                .find(|p| p.source_table == table)
                .map(|p| p.destination_collection.clone())
        };
        assert_eq!(find("emp_master"), Some("employees".to_string()));
        assert_eq!(find("dept_info"), Some("departments".to_string()));
        assert_eq!(find("locations"), Some("locations".to_string()));
    }

    // --- end-to-end matching: helper ---

    /// Run the full core pipeline on the embedded schemas and return the
    /// destination assigned to `source_field` in `source_table`, if any.
    fn mapped_destination(source_table: &str, source_field: &str) -> Option<String> {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        for table in &results {
            if table.source_table == source_table {
                for mapping in &table.mappings {
                    if mapping.source_field == source_field {
                        return Some(mapping.destination_field.clone());
                    }
                }
            }
        }
        None
    }

    // --- end-to-end matching: key mappings from the assignment ---

    #[test]
    fn smap_e2e_name_fields_map_into_fullname_subdocument() {
        assert_eq!(
            mapped_destination("emp_master", "f_name"),
            Some("fullName.firstName".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "l_name"),
            Some("fullName.lastName".to_string())
        );
    }

    #[test]
    fn smap_e2e_primary_keys_map_to_mongo_id() {
        assert_eq!(
            mapped_destination("emp_master", "emp_id"),
            Some("_id".to_string())
        );
        assert_eq!(
            mapped_destination("dept_info", "dept_id"),
            Some("_id".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "loc_id"),
            Some("_id".to_string())
        );
    }

    #[test]
    fn smap_e2e_semantic_synonyms_hire_and_term() {
        assert_eq!(
            mapped_destination("emp_master", "hire_dt"),
            Some("employment.startDate".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "term_dt"),
            Some("employment.endDate".to_string())
        );
    }

    #[test]
    fn smap_e2e_status_code_fields() {
        assert_eq!(
            mapped_destination("emp_master", "rec_stat"),
            Some("employment.status".to_string())
        );
        // Domain rule: CHAR(1) A/I status -> Boolean isActive.
        assert_eq!(
            mapped_destination("dept_info", "dept_stat"),
            Some("isActive".to_string())
        );
    }

    #[test]
    fn smap_e2e_remaining_employee_fields() {
        assert_eq!(
            mapped_destination("emp_master", "emp_cd"),
            Some("employeeCode".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "dept_id"),
            Some("department.departmentId".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "mgr_emp_id"),
            Some("employment.managerId".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "job_lvl_cd"),
            Some("employment.jobLevel".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "base_sal"),
            Some("compensation.baseSalary".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "sal_currency"),
            Some("compensation.currency".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "work_email"),
            Some("contact.email".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "work_phone"),
            Some("contact.phone".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "office_loc_id"),
            Some("location.locationId".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "is_remote"),
            Some("employment.isRemote".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "created_ts"),
            Some("meta.createdAt".to_string())
        );
        assert_eq!(
            mapped_destination("emp_master", "updated_ts"),
            Some("meta.updatedAt".to_string())
        );
    }

    #[test]
    fn smap_e2e_locations_fields() {
        assert_eq!(
            mapped_destination("locations", "loc_cd"),
            Some("code".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "loc_nm"),
            Some("name".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "city"),
            Some("city".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "state_prov"),
            Some("stateOrProvince".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "country_cd"),
            Some("country".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "postal_cd"),
            Some("postalCode".to_string())
        );
        assert_eq!(
            mapped_destination("locations", "tz_cd"),
            Some("timezone".to_string())
        );
    }

    #[test]
    fn smap_e2e_departments_fields() {
        assert_eq!(
            mapped_destination("dept_info", "dept_cd"),
            Some("code".to_string())
        );
        assert_eq!(
            mapped_destination("dept_info", "dept_nm"),
            Some("name".to_string())
        );
        assert_eq!(
            mapped_destination("dept_info", "parent_dept_id"),
            Some("parentDepartmentId".to_string())
        );
        assert_eq!(
            mapped_destination("dept_info", "dept_head_id"),
            Some("headEmployeeId".to_string())
        );
        assert_eq!(
            mapped_destination("dept_info", "cost_ctr_cd"),
            Some("costCenterCode".to_string())
        );
    }

    #[test]
    fn smap_e2e_dob_is_unmapped_source_field() {
        // Dataset B has no birth-date field; dob must land in
        // unmapped_source_fields, not be force-matched to anything.
        assert_eq!(mapped_destination("emp_master", "dob"), None);
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let emp_table = results
            .iter()
            .find(|t| t.source_table == "emp_master")
            .expect("emp_master pair must exist");
        assert!(
            emp_table
                .unmapped_source_fields
                .contains(&"dob".to_string())
        );
    }

    // --- notes / transforms ---

    #[test]
    fn smap_notes_rec_stat_enum_transform() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let emp_table = results
            .iter()
            .find(|t| t.source_table == "emp_master")
            .expect("emp_master pair must exist");
        let rec_stat = emp_table
            .mappings
            .iter()
            .find(|m| m.source_field == "rec_stat")
            .expect("rec_stat mapping must exist");
        let notes = rec_stat
            .notes
            .as_deref()
            .expect("rec_stat must carry a transform note");
        assert!(notes.contains("A -> active"));
        assert!(notes.contains("I -> inactive"));
        assert!(notes.contains("T -> terminated"));
    }

    #[test]
    fn smap_notes_dept_stat_boolean_transform() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let dept_table = results
            .iter()
            .find(|t| t.source_table == "dept_info")
            .expect("dept_info pair must exist");
        let dept_stat = dept_table
            .mappings
            .iter()
            .find(|m| m.source_field == "dept_stat")
            .expect("dept_stat mapping must exist");
        let notes = dept_stat
            .notes
            .as_deref()
            .expect("dept_stat must carry a transform note");
        assert!(notes.contains("A -> true"));
        assert!(notes.contains("I -> false"));
    }

    #[test]
    fn smap_notes_is_remote_boolean_transform() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let emp_table = results
            .iter()
            .find(|t| t.source_table == "emp_master")
            .expect("emp_master pair must exist");
        let is_remote = emp_table
            .mappings
            .iter()
            .find(|m| m.source_field == "is_remote")
            .expect("is_remote mapping must exist");
        assert_eq!(
            is_remote.type_transform,
            "TINYINT(1) -> Boolean (nested path)"
        );
        let notes = is_remote
            .notes
            .as_deref()
            .expect("is_remote must carry a transform note");
        assert!(notes.contains("0 -> false"));
        assert!(notes.contains("1 -> true"));
    }

    // --- JSON rendering ---

    #[test]
    fn smap_json_escape_special_characters() {
        assert_eq!(json_escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(json_escape("line1\nline2"), "line1\\nline2");
        assert_eq!(json_escape("plain"), "plain");
    }

    #[test]
    fn smap_render_mapping_json_contains_required_top_level_keys() {
        let (sql_fields, _s) = parse_schema_text(SQL_SCHEMA_TEXT, "tables");
        let (mongo_fields, _m) = parse_schema_text(MONGO_SCHEMA_TEXT, "collections");
        let results = build_all_mappings(&sql_fields, &mongo_fields)
            .expect("pipeline must produce results on known-good embedded input");
        let rendered = render_mapping_json(&results, "2026-01-01T00:00:00Z");
        assert!(rendered.contains("\"mapping_version\": \"1.0\""));
        assert!(rendered.contains("\"source\": \"legacy_hrm (MySQL)\""));
        assert!(rendered.contains("\"destination\": \"people_platform (MongoDB)\""));
        assert!(rendered.contains("\"generated_at\": \"2026-01-01T00:00:00Z\""));
        assert!(rendered.contains("\"unmapped_source_fields\""));
        assert!(rendered.contains("\"unmapped_destination_fields\""));
    }

    // --- timestamps ---

    #[test]
    fn smap_civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // 2024-01-01
        assert_eq!(civil_from_days(11_016), (2000, 2, 29)); // leap day 2000
    }

    // --- error codes ---

    #[test]
    fn smap_error_codes_are_stable() {
        // Error-Code-Table is APPEND-ONLY; these values must never change.
        assert_eq!(SmapError::ReadSqlSchemaFailed.code(), 101);
        assert_eq!(SmapError::ReadMongoSchemaFailed.code(), 102);
        assert_eq!(SmapError::WriteMappingFailed.code(), 103);
        assert_eq!(SmapError::WriteMetadataFailed.code(), 104);
        assert_eq!(SmapError::WriteLogFailed.code(), 105);
        assert_eq!(SmapError::SqlParseNoFields.code(), 201);
        assert_eq!(SmapError::MongoParseNoFields.code(), 202);
        assert_eq!(SmapError::NoTablePairs.code(), 301);
        assert_eq!(SmapError::ClockBeforeEpoch.code(), 401);
    }

    // --- parser input-validation behavior (skip, never panic) ---

    #[test]
    fn smap_parser_tolerates_malformed_lines() {
        let malformed = "garbage line\n\"unclosed: INT\n}}}\n{\n\"x\": {\n\"y\": INT\n}\n";
        // Must not panic; malformed content is counted as skipped.
        let (fields, skipped) = parse_schema_text(malformed, "tables");
        assert!(skipped >= 2);
        // "y" is under "x" but "x" is not under a "tables" key -> skipped.
        assert!(fields.is_empty());
    }

    #[test]
    fn smap_parser_empty_input_yields_zero_fields() {
        let (fields, _skipped) = parse_schema_text("", "tables");
        assert!(fields.is_empty());
    }
}

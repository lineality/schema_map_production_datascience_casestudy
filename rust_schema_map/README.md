DB-Mapping Fragment Accumulator:  Rust
Geoffrey Gordon Ashbrook 2026.07.09th

(note: Full assignment instructions in doc.)

Rule: "You cannot pass both schemas to an LLM in a single prompt and receive a finished mapping."


# Overall Plan:
- Rust
- input two db~schema files
- Produce a map json


A .py LLM version (with dubious practical applications) exists, but can a production Rust version be made? Let's explore.

### Rust & Levinshtein Distance et al
This task is a schema-matching problem, which is much more structure-intensive than many very-unstructured-input uses where an LLM may be needed.  Schema-matching has decades of history and literature for GOFAI/statistical techniques (Levenshtein, token-set similarity, structural/type compatibility, abbreviation dictionaries) that are more production-friendly than LLMs. 

E.g. see https://datamade.us/blog/schema-matching/ 

A Rust implementation, namely a vanilla-rust (standard-library-only) approach may have significant advantage over python:
- versatility of deployability
- slimmer: efficiency
- faster
- safter
- more robust atomics, concurrency, parallelization
- better error tracking and auditing
- explainability
- predictability
- maintainability
- memory-safe, various benefits possibly including strict handling of secure values
- long term asset: python LLM code is highly unlikely to still run in 50 years, and probably not even 5 years (possibly not even one year), whereas this Rust code would be a long term asset, likely to run indefinitely.
- no third party dependencies
- strict production-grade mode and case/error handling
- more meaningful confidence-score

Note: It took significantly longer to design and make the Rube-Goldberg LLM-fake solution-in-search-of-a-problem toy, compared with the Rust-NLP-GOFAI system that does have the potential to be useful in production and general practical use.


# Program details:
- see rust rules
- see mode and case handling! (important)

DEFAULT_OUTPUT_MAPPING_JSON_PATH = f"schema_mapping_output_{readable_timestamp}.json"

DEFAULT_OUTPUT_METADATA_REPORT_PATH = f"schema_mapper_metadata_report_{readable_timestamp}.json"

DEFAULT_RUN_LOG_FILE_PATH = f"schema_mapper_log_{readable_timestamp}.log"

input paths can be hardcoded for now:

SQL_SCHEMA_FILE_PATH=sql_schema_legacy_hrm.txt

MONGO_SCHEMA_FILE_PATH=mongo_schema_people_platform.txt

It should be possible to write a basic Json without serd-crate.

- Single main.rs
- heap-use not a strict issue, but pre-allocating buffers e.g. for known line-length, is reasonable here
- write result to file

File format:

Input files are pseudo-JSON with inline comments, not real valid JSON. 
parser: a line-oriented parser tailored to this known file format (columns: name, type, annotations, -- comments)


the format is obviously not an exact match
reasoning, notes, and confidence can be defined-ish here.

e.g. "reasoning" can be an aggregate log of what 'filter' caught that match, which make sense: why was this included, because XYZ

note: if there are specific edge cases known, a note could go there (or maybe not used). This could also work with 'confidence' e.g. 'exact match' may be good to note.

confidence is not always relevant, but in cases of exact match, or a slight string-distance, confidence and notes could record this.

if the match was custom-helped based on domain-knowledge, a note could be made for that too, and reasoning note perhaps.

exit behavior: 0 or 1


fancy formatting such as Buffy-format likely not needed here.

...

fedora:~/code/schema_map/rust_schema_map/schema_map_rust$ time cargo run --release
    Finished `release` profile [optimized] target(s) in 0.00s
     Running `target/release/schema_map_rust`

e.g. 3-17 milliseconds

real	0m0.017s
user	0m0.009s
sys	0m0.009s
/
real	0m0.017s
user	0m0.015s
sys	0m0.003s











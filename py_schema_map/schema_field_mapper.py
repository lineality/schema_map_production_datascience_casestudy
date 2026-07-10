"""
schema_field_mapper.py

DB-Mapping Fragment Accumulator v1 (MVP-1)

Purpose:
    Maps every field in a source MySQL schema (legacy_hrm) to its semantically
    equivalent field in a destination MongoDB schema (people_platform),
    producing a single JSON mapping document plus a separate run-metadata
    report file.

Architecture (per agreed scope):
    Part 1 (deterministic Python, no LLM):
        - Read two .txt schema files (paths from .env, input() fallback).
        - Chop the SQL schema text into per-table string fragments.
        - Build field-name dictionaries for both schemas. MongoDB field
          names are stored as flattened dot-paths (e.g. fullName.firstName)
          and are used to validate LLM-returned destination_field values.
        - Extract report metadata (source label, destination label).
    Part 2 (LLM loop):
        - For each SQL table fragment: one LLM call containing that single
          table fragment plus the ENTIRE MongoDB schema. This respects the
          assignment constraint: both full schemas are never in one prompt.
        - LLM output is structured via pydantic models using the official
          mistralai library's structured-output support.
        - Guard-rails: every LLM-returned field name is validated against
          the deterministic field dictionaries. Invalid output is logged
          and retried up to a capped number of attempts, then hard error.
    Part 3 (output):
        - Assemble and write the final mapping JSON (assignment format,
          exactly), print it to the terminal, and write a separate
          metadata/issues report (timings, call counts, errors, collisions).

Environment variables (via .env / python-dotenv):
    SQL_SCHEMA_FILE_PATH            path to the MySQL schema .txt file
    MONGO_SCHEMA_FILE_PATH          path to the MongoDB schema .txt file
    MISTRAL_API_KEY                 Mistral API key (required, no fallback)
    MISTRAL_MODEL_NAME              e.g. "mistral-small-latest"
    OUTPUT_MAPPING_JSON_PATH        optional, default "schema_mapping_output.json"
    OUTPUT_METADATA_REPORT_PATH     optional, default "run_metadata_report.json"
    MAX_LLM_API_RETRY_ATTEMPTS      optional, default 10
    MAX_LLM_VALIDATION_RETRY_ATTEMPTS  optional, default 10

Out of scope for MVP-1 (per agreed scope):
    - Any-DB generalization ('table' / 'collection' terminology is fixed).
    - Arbitrary input formats (input is the assignment's pseudo-format).
    - Providers other than Mistral (design allows later substitution).
    - Per-provider token accounting.
"""

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv
from mistralai.client import Mistral
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Default values (overridable via .env; no environment values hard-coded)
# ---------------------------------------------------------------------------

# Get the current date and time
timestampnow = datetime.now()

# Option 1: Standard clean format (2026-07-09 16:14:30)
readable_timestamp = timestampnow.strftime("%Y_%m_%d__%H%M%S")

DEFAULT_OUTPUT_MAPPING_JSON_PATH: str = f"schema_mapping_output_{readable_timestamp}.json"
DEFAULT_OUTPUT_METADATA_REPORT_PATH: str = f"schema_mapper_metadata_report_{readable_timestamp}.json"
DEFAULT_RUN_LOG_FILE_PATH: str = f"schema_mapper_log_{readable_timestamp}.log"
DEFAULT_MAX_LLM_API_RETRY_ATTEMPTS: int = 10
DEFAULT_MAX_LLM_VALIDATION_RETRY_ATTEMPTS: int = 10

MAPPING_VERSION_STRING: str = "1.0"

def append_timestamped_log_entry(
        run_log_file_path: str, severity_label: str, message_text: str) -> None:
    """
    Append one timestamped entry to the plain-text run log file, and echo
    it to stderr (warnings/errors) or stdout (info).

    The log file is opened in append mode per entry so that every event is
    on disk immediately; a hard process death mid-run therefore does not
    lose earlier entries (unlike the end-of-run metadata report).

    Args:
        run_log_file_path: Path to the append-mode run log file.
        severity_label: One of "info", "warning", "error", "fatal".
        message_text: The event description.

    Note:
        A failure to write the log file is reported to stderr but is NOT
        raised: logging failure must never mask or abort the pipeline's
        substantive work. This is deliberate and is the only place in the
        application where an exception is intentionally not propagated.
    """
    timestamped_line = (
        f"{datetime.now(timezone.utc).isoformat()} "
        f"[{severity_label}] {message_text}"
    )
    try:
        with open(run_log_file_path, "a", encoding="utf-8") as log_file_handle:
            log_file_handle.write(timestamped_line + "\n")
    except OSError as log_write_error:
        print(
            f"[warning] Could not write to run log file "
            f"'{run_log_file_path}': {log_write_error}",
            file=sys.stderr,
        )
    # Echo to the terminal as before.
    output_stream = sys.stderr if severity_label != "info" else sys.stdout
    print(f"[{severity_label}] {message_text}", file=output_stream)

# ---------------------------------------------------------------------------
# Pydantic models for LLM structured output
# (Class-based by necessity: the mistralai structured-output feature
#  requires pydantic BaseModel subclasses. All other code is functional.)
# ---------------------------------------------------------------------------

class LlmSingleFieldMapping(BaseModel):
    """
    One source-field-to-destination-field mapping, as returned by the LLM.

    Attributes:
        source_field: Field name exactly as it appears in the SQL table.
        destination_field: Dot-path within the destination collection,
            e.g. "fullName.firstName". Does NOT include the collection name.
        type_transform: Plain-text type conversion, e.g. "TINYINT(1) -> Boolean".
        confidence: Float in [0.0, 1.0] expressing the LLM's stated confidence.
        reasoning: One plain-English sentence explaining the match.
        notes: Any value-transform logic required, or None.
    """
    source_field: str
    destination_field: str
    type_transform: str
    confidence: float
    reasoning: str
    notes: str | None


class LlmSingleTableMapping(BaseModel):
    """
    Complete mapping object for one SQL table, as returned by the LLM.

    Attributes:
        source_table: SQL table name exactly as given in the prompt.
        destination_collection: Name of the matched MongoDB collection.
        confidence: Table-level match confidence in [0.0, 1.0].
        reasoning: One plain-English sentence for the table-level match.
        field_mappings: One entry per mapped source field.
        unmapped_source_fields: Source fields with no destination equivalent.
        unmapped_destination_fields: Destination dot-paths (within the chosen
            collection) that no source field of this table maps to.
    """
    source_table: str
    destination_collection: str
    confidence: float
    reasoning: str
    field_mappings: list[LlmSingleFieldMapping]
    unmapped_source_fields: list[str]
    unmapped_destination_fields: list[str]


# ---------------------------------------------------------------------------
# Join-Derived Fields
# ---------------------------------------------------------------------------

class LlmJoinDerivedFieldProposal(BaseModel):
    """
    One proposed join-derived mapping: a source field whose value would be
    copied (denormalized) into an embedded destination field via a join.

    Attributes:
        destination_field: Dot-path of the embedded destination field,
            e.g. "department.name".
        source_field: Field in the join-target SQL table, e.g. "dept_nm".
        type_transform: Plain-text type conversion.
        confidence: Float in [0.0, 1.0].
        reasoning: One plain-English sentence explaining the match.
        notes: Any value-transform logic required, or None.
    """
    destination_field: str
    source_field: str
    type_transform: str
    confidence: float
    reasoning: str
    notes: str | None


class LlmJoinDerivedMappingSet(BaseModel):
    """
    LLM response for one join group: proposals for the unmapped embedded
    fields, plus any fields the model judges unresolvable from this table.

    Attributes:
        proposals: One entry per resolvable destination field.
        unresolvable_destination_fields: Dot-paths from the provided list
            that have no equivalent in the join-target table.
    """
    proposals: list[LlmJoinDerivedFieldProposal]
    unresolvable_destination_fields: list[str]

def call_mistral_for_join_derived_mapping_set(
        mistral_client: Mistral,
        mistral_model_name: str,
        prompt_text: str,
        max_api_retry_attempts: int,
        run_metadata_accumulator: dict) -> LlmJoinDerivedMappingSet:
    """
    Call the Mistral API with structured (pydantic) output for one
    join-derived mapping group, retrying on API/transport failure up to
    max_api_retry_attempts.

    This deliberately mirrors call_mistral_for_single_table_mapping; the
    duplication is accepted in MVP-1 in favor of explicit, separately
    readable call paths. (A shared generic caller parameterized on the
    response model is a reasonable future refactor.)

    Args:
        mistral_client: An initialized mistralai Mistral client.
        mistral_model_name: Model identifier, e.g. "mistral-medium-2508".
        prompt_text: The complete join-group prompt.
        max_api_retry_attempts: Hard cap on API attempts before raising.
        run_metadata_accumulator: Mutable dict used for run reporting;
            this function appends per-call durations and error records to
            it (keys: "llm_call_durations_seconds", "llm_call_count",
            "errors").

    Returns:
        A validated LlmJoinDerivedMappingSet instance parsed by pydantic.

    Raises:
        RuntimeError: If all API attempts fail, or if the API returns a
            response with no parsed structured content.
    """
    most_recent_api_error: Exception | None = None

    for api_attempt_number in range(1, max_api_retry_attempts + 1):
        call_start_time = time.monotonic()
        try:
            api_response = mistral_client.chat.parse(
                model=mistral_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a database schema-migration analyst. "
                            "You produce precise field mappings using only "
                            "the field names provided. You never invent "
                            "field names."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                response_format=LlmJoinDerivedMappingSet,
                temperature=0.0,
            )
            call_duration_seconds = time.monotonic() - call_start_time
            run_metadata_accumulator["llm_call_durations_seconds"].append(
                call_duration_seconds)
            run_metadata_accumulator["llm_call_count"] += 1

            parsed_mapping_set = api_response.choices[0].message.parsed
            if parsed_mapping_set is None:
                raise RuntimeError(
                    "Mistral API returned a response but the parsed "
                    "structured content is None (structured-output parse "
                    "failure on the service side)."
                )
            return parsed_mapping_set

        except Exception as api_call_error:
            call_duration_seconds = time.monotonic() - call_start_time
            most_recent_api_error = api_call_error
            error_record = (
                f"Join-derived API attempt "
                f"{api_attempt_number}/{max_api_retry_attempts} failed "
                f"after {call_duration_seconds:.2f}s: "
                f"{type(api_call_error).__name__}: {api_call_error}"
            )
            print(f"[warning] {error_record}", file=sys.stderr)
            run_metadata_accumulator["errors"].append(error_record)
            time.sleep(min(2.0 * api_attempt_number, 15.0))

    raise RuntimeError(
        f"All {max_api_retry_attempts} Mistral API attempts failed for a "
        f"join-derived mapping group. Most recent error: "
        f"{type(most_recent_api_error).__name__}: {most_recent_api_error}"
    )


def validate_join_derived_mapping_set(
        join_derived_mapping_set: LlmJoinDerivedMappingSet,
        join_source_table_field_names: list[str],
        expected_destination_dot_paths: list[str]) -> list[str]:
    """
    Validate one LLM-produced join-derived mapping set against the
    deterministic field dictionaries. Anti-hallucination guard-rail for
    the join-derived pass.

    Checks performed:
        1. Every proposal's source_field exists in the join-source table.
        2. Every proposal's destination_field is one of the dot-paths
           this group was asked about.
        3. Every unresolvable_destination_fields entry is one of the
           dot-paths this group was asked about.
        4. Coverage: every requested dot-path appears in proposals or in
           unresolvable_destination_fields.
        5. Disjointness: no dot-path appears in both proposals and
           unresolvable_destination_fields.
        6. Every confidence value is within [0.0, 1.0].

    Args:
        join_derived_mapping_set: The LLM output to validate.
        join_source_table_field_names: Deterministically-parsed field
            names of the join-source table.
        expected_destination_dot_paths: The dot-paths this group's prompt
            asked the model to resolve.

    Returns:
        A list of human-readable issue messages. Empty list means the
        mapping set passed all checks.
    """
    validation_issue_messages: list[str] = []
    source_field_name_set = set(join_source_table_field_names)
    expected_dot_path_set = set(expected_destination_dot_paths)

    # Checks 1, 2, 6: per-proposal integrity.
    for proposal in join_derived_mapping_set.proposals:
        if proposal.source_field not in source_field_name_set:
            validation_issue_messages.append(
                f"proposals contains source_field '{proposal.source_field}' "
                "which does not exist in the join-source table."
            )
        if proposal.destination_field not in expected_dot_path_set:
            validation_issue_messages.append(
                f"proposals contains destination_field "
                f"'{proposal.destination_field}' which is not one of the "
                f"requested dot-paths: {sorted(expected_dot_path_set)}."
            )
        if not (0.0 <= proposal.confidence <= 1.0):
            validation_issue_messages.append(
                f"confidence {proposal.confidence} for destination_field "
                f"'{proposal.destination_field}' is outside [0.0, 1.0]."
            )

    # Check 3: unresolvable entries must be from the requested list.
    for unresolvable_dot_path in \
            join_derived_mapping_set.unresolvable_destination_fields:
        if unresolvable_dot_path not in expected_dot_path_set:
            validation_issue_messages.append(
                f"unresolvable_destination_fields contains "
                f"'{unresolvable_dot_path}' which is not one of the "
                f"requested dot-paths."
            )

    # Checks 4 and 5: coverage and disjointness.
    proposed_dot_path_set = {
        proposal.destination_field
        for proposal in join_derived_mapping_set.proposals
    }
    unresolvable_dot_path_set = set(
        join_derived_mapping_set.unresolvable_destination_fields)
    missing_dot_paths = expected_dot_path_set - (
        proposed_dot_path_set | unresolvable_dot_path_set)
    if missing_dot_paths:
        validation_issue_messages.append(
            f"These requested dot-paths appear in neither proposals nor "
            f"unresolvable_destination_fields: {sorted(missing_dot_paths)}. "
            "Every requested dot-path must be covered."
        )
    contradictory_dot_paths = proposed_dot_path_set & unresolvable_dot_path_set
    if contradictory_dot_paths:
        validation_issue_messages.append(
            f"These dot-paths appear in BOTH proposals and "
            f"unresolvable_destination_fields, which is contradictory: "
            f"{sorted(contradictory_dot_paths)}."
        )

    return validation_issue_messages


def parse_foreign_key_edges_from_sql_schema(
        sql_table_name_to_body: dict[str, str]) -> dict[str, dict[str, tuple[str, str]]]:
    """
    Parse foreign-key annotations from all SQL table bodies.

    Recognizes lines of the assignment format containing
    'FK -> <table>.<field>', e.g.:
        "dept_id":       INT            FK -> dept_info.dept_id

    Args:
        sql_table_name_to_body: table name -> inner body text.

    Returns:
        Nested dict: table name -> {source field name ->
        (referenced table name, referenced field name)}. Tables with no
        FK fields map to an empty dict.
    """
    foreign_key_line_pattern = re.compile(
        r'^\s*"([A-Za-z0-9_]+)"\s*:.*FK\s*->\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)',
        re.MULTILINE,
    )
    table_to_foreign_key_edges: dict[str, dict[str, tuple[str, str]]] = {}
    for table_name, table_body in sql_table_name_to_body.items():
        table_to_foreign_key_edges[table_name] = {
            field_name: (referenced_table, referenced_field)
            for field_name, referenced_table, referenced_field
            in foreign_key_line_pattern.findall(table_body)
        }
    return table_to_foreign_key_edges


def obtain_validated_join_derived_mapping_set(
        mistral_client: Mistral,
        application_configuration: dict,
        join_group_record: dict,
        join_source_table_body: str,
        join_source_table_field_names: list[str],
        run_metadata_accumulator: dict) -> LlmJoinDerivedMappingSet:
    """
    Run the call-then-validate loop for one join group until the mapping
    set passes validation, or hard-error at the validation retry cap.

    On each retry, the validation issues from the previous attempt are
    appended to the prompt so the model can self-correct.

    Args:
        mistral_client: An initialized mistralai Mistral client.
        application_configuration: Config dict from
            load_application_configuration().
        join_group_record: One record from
            identify_join_groups_for_unmapped_fields.
        join_source_table_body: Inner body text of the join-source table.
        join_source_table_field_names: Deterministically-parsed field
            names of the join-source table.
        run_metadata_accumulator: Mutable dict for run reporting; this
            function appends validation-issue records to it.

    Returns:
        A LlmJoinDerivedMappingSet that passed all validation checks.

    Raises:
        RuntimeError: If the validation retry cap is exhausted.
    """
    max_validation_attempts = application_configuration[
        "max_llm_validation_retry_attempts"]
    base_prompt_text = build_join_derived_mapping_prompt(
        join_group_record=join_group_record,
        join_source_table_body=join_source_table_body,
    )
    previous_attempt_issue_messages: list[str] = []
    group_description = (
        f"{join_group_record['destination_collection']}."
        f"{join_group_record['embedded_parent_path']} "
        f"(via {join_group_record['join_source_table']})"
    )

    for validation_attempt_number in range(1, max_validation_attempts + 1):
        # Append retry feedback to the base prompt (the prompt builder
        # itself is not modified; feedback is a retry-only concern).
        prompt_text = base_prompt_text
        if previous_attempt_issue_messages:
            rendered_issues = "\n".join(
                f"- {issue}" for issue in previous_attempt_issue_messages)
            prompt_text += (
                "\n\n# Issues found in your previous attempt "
                f"(fix ALL of these):\n{rendered_issues}"
            )

        join_derived_mapping_set = call_mistral_for_join_derived_mapping_set(
            mistral_client=mistral_client,
            mistral_model_name=application_configuration["mistral_model_name"],
            prompt_text=prompt_text,
            max_api_retry_attempts=application_configuration[
                "max_llm_api_retry_attempts"],
            run_metadata_accumulator=run_metadata_accumulator,
        )
        validation_issue_messages = validate_join_derived_mapping_set(
            join_derived_mapping_set=join_derived_mapping_set,
            join_source_table_field_names=join_source_table_field_names,
            expected_destination_dot_paths=join_group_record[
                "unmapped_destination_dot_paths"],
        )
        if not validation_issue_messages:
            print(
                f"[info] Join group {group_description}: mapping set passed "
                f"validation on attempt {validation_attempt_number}."
            )
            return join_derived_mapping_set

        for issue_message in validation_issue_messages:
            issue_record = (
                f"Join group {group_description} validation attempt "
                f"{validation_attempt_number}/{max_validation_attempts}: "
                f"{issue_message}"
            )
            print(f"[warning] {issue_record}", file=sys.stderr)
            run_metadata_accumulator["validation_issues"].append(issue_record)
        previous_attempt_issue_messages = validation_issue_messages

    raise RuntimeError(
        f"Join group {group_description}: LLM output failed validation on "
        f"all {max_validation_attempts} attempts. Final issues: "
        f"{previous_attempt_issue_messages}"
    )


def identify_join_groups_for_unmapped_fields(
        validated_table_mappings: list[LlmSingleTableMapping],
        globally_unmapped_destination_fields: dict[str, list[str]],
        table_to_foreign_key_edges: dict[str, dict[str, tuple[str, str]]]) -> list[dict]:
    """
    Deterministically attach globally-unmapped embedded destination fields
    to the SQL table that a join would reach.

    Method: a 'join anchor' is an existing field mapping whose source
    field carries an FK and whose destination dot-path sits inside an
    embedded sub-document (e.g. emp_master.dept_id ->
    employees.department.departmentId, FK -> dept_info). Every globally-
    unmapped field under that same sub-document parent is then attributed
    to the FK's target table as its candidate join source.

    Args:
        validated_table_mappings: All validated per-table mapping objects.
        globally_unmapped_destination_fields: collection name -> dot-paths
            never mapped by any table (from
            compute_globally_unmapped_destination_fields).
        table_to_foreign_key_edges: Output of
            parse_foreign_key_edges_from_sql_schema.

    Returns:
        List of join-group dicts with keys: "destination_collection",
        "embedded_parent_path", "join_source_table", "join_via" (a
        human-readable FK description), and
        "unmapped_destination_dot_paths". Unmapped fields whose parent has
        no join anchor are omitted here (they remain visible in
        globally_unmapped_destination_fields).
    """
    # Build anchor registry: (collection, embedded parent) -> (table, via).
    anchor_registry: dict[tuple[str, str], tuple[str, str]] = {}
    for table_mapping in validated_table_mappings:
        fk_edges_for_table = table_to_foreign_key_edges.get(
            table_mapping.source_table, {})
        for field_mapping_entry in table_mapping.field_mappings:
            if field_mapping_entry.source_field not in fk_edges_for_table:
                continue
            if "." not in field_mapping_entry.destination_field:
                continue  # destination is not inside an embedded sub-document
            embedded_parent_path = field_mapping_entry.destination_field.rsplit(
                ".", 1)[0]
            referenced_table, referenced_field = fk_edges_for_table[
                field_mapping_entry.source_field]
            join_via_description = (
                f"{table_mapping.source_table}."
                f"{field_mapping_entry.source_field} -> "
                f"{referenced_table}.{referenced_field}"
            )
            anchor_registry[(table_mapping.destination_collection,
                             embedded_parent_path)] = (
                referenced_table, join_via_description)

    # Group unmapped fields under their anchored parents.
    join_group_records: list[dict] = []
    for collection_name, unmapped_dot_paths in \
            globally_unmapped_destination_fields.items():
        parent_to_dot_paths: dict[str, list[str]] = {}
        for dot_path in unmapped_dot_paths:
            if "." not in dot_path:
                continue  # top-level field, no embedded parent to anchor
            parent_to_dot_paths.setdefault(
                dot_path.rsplit(".", 1)[0], []).append(dot_path)
        for embedded_parent_path, grouped_dot_paths in \
                parent_to_dot_paths.items():
            anchor = anchor_registry.get(
                (collection_name, embedded_parent_path))
            if anchor is None:
                continue  # no FK anchor found; stays globally-unmapped only
            join_source_table, join_via_description = anchor
            join_group_records.append({
                "destination_collection": collection_name,
                "embedded_parent_path": embedded_parent_path,
                "join_source_table": join_source_table,
                "join_via": join_via_description,
                "unmapped_destination_dot_paths": sorted(grouped_dot_paths),
            })
    return join_group_records


def build_join_derived_mapping_prompt(
        join_group_record: dict, join_source_table_body: str) -> str:
    """
    Build the prompt for one join group: one SQL table fragment plus the
    short list of unmapped embedded destination dot-paths.

    Constraint compliance note: this prompt contains one SQL table
    fragment and a handful of destination dot-paths — far less than one
    full schema of either kind.

    Args:
        join_group_record: One record from
            identify_join_groups_for_unmapped_fields.
        join_source_table_body: Inner body text of the join-source table.

    Returns:
        The complete prompt string.
    """
    rendered_dot_paths = "\n".join(
        f"- {dot_path}"
        for dot_path in join_group_record["unmapped_destination_dot_paths"])
    return f"""# Join-derived (denormalized) field mapping task

During a MySQL-to-MongoDB migration, the destination collection
'{join_group_record["destination_collection"]}' embeds a copy of related
data under '{join_group_record["embedded_parent_path"]}'. The join path is:
{join_group_record["join_via"]}

## Source table '{join_group_record["join_source_table"]}' (MySQL):
"{join_group_record["join_source_table"]}": {{
{join_source_table_body}
}}

## Destination fields needing a source (dot-paths, currently unmapped):
{rendered_dot_paths}

## Requirements
1. For each destination dot-path, propose the source field from the table
   above whose value would be copied in via the join, or list it in
   unresolvable_destination_fields if no equivalent exists.
2. source_field must be a field of '{join_group_record["join_source_table"]}'.
3. destination_field must be exactly one of the dot-paths listed above.
4. confidence: a float between 0.0 and 1.0. reasoning: one sentence.
   notes: value-transform logic if required, otherwise null.

Return only the structured mapping object."""

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_application_configuration() -> dict:
    """
    Load all configuration from the .env file and the process environment.

    File paths missing from the environment are requested from the user
    interactively via input() (per agreed scope). A missing API key is a
    hard error: secrets are never requested interactively.

    Returns:
        A dict with keys:
            sql_schema_file_path (str)
            mongo_schema_file_path (str)
            mistral_api_key (str)
            mistral_model_name (str)
            output_mapping_json_path (str)
            output_metadata_report_path (str)
            max_llm_api_retry_attempts (int)
            max_llm_validation_retry_attempts (int)

    Raises:
        ValueError: If MISTRAL_API_KEY or MISTRAL_MODEL_NAME is absent,
            or if a retry-limit value cannot be parsed as an integer.
    """
    # Load .env into the process environment (no-op if file absent).
    load_dotenv()

    sql_schema_file_path = os.getenv("SQL_SCHEMA_FILE_PATH")
    if not sql_schema_file_path:
        # Interactive fallback for file paths only, per agreed scope.
        sql_schema_file_path = input(
            "SQL_SCHEMA_FILE_PATH not set in .env. "
            "Enter path to the MySQL schema .txt file: "
        ).strip()

    mongo_schema_file_path = os.getenv("MONGO_SCHEMA_FILE_PATH")
    if not mongo_schema_file_path:
        mongo_schema_file_path = input(
            "MONGO_SCHEMA_FILE_PATH not set in .env. "
            "Enter path to the MongoDB schema .txt file: "
        ).strip()

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not set in the environment or .env file. "
            "This is required and will not be requested interactively. "
            "Add MISTRAL_API_KEY=<your key> to the .env file."
        )

    mistral_model_name = os.getenv("MISTRAL_MODEL_NAME")
    if not mistral_model_name:
        raise ValueError(
            "MISTRAL_MODEL_NAME is not set in the environment or .env file. "
            "Add e.g. MISTRAL_MODEL_NAME=mistral-small-latest to the .env file."
        )

    try:
        max_llm_api_retry_attempts = int(
            os.getenv("MAX_LLM_API_RETRY_ATTEMPTS",
                      str(DEFAULT_MAX_LLM_API_RETRY_ATTEMPTS))
        )
        max_llm_validation_retry_attempts = int(
            os.getenv("MAX_LLM_VALIDATION_RETRY_ATTEMPTS",
                      str(DEFAULT_MAX_LLM_VALIDATION_RETRY_ATTEMPTS))
        )
    except ValueError as integer_parse_error:
        raise ValueError(
            "MAX_LLM_API_RETRY_ATTEMPTS and MAX_LLM_VALIDATION_RETRY_ATTEMPTS "
            "must be integers if set in .env. "
            f"Parse failure detail: {integer_parse_error}"
        ) from integer_parse_error

    return {
        "sql_schema_file_path": sql_schema_file_path,
        "mongo_schema_file_path": mongo_schema_file_path,
        "mistral_api_key": mistral_api_key,
        "mistral_model_name": mistral_model_name,
        "output_mapping_json_path": os.getenv(
            "OUTPUT_MAPPING_JSON_PATH", DEFAULT_OUTPUT_MAPPING_JSON_PATH),
        "output_metadata_report_path": os.getenv(
            "OUTPUT_METADATA_REPORT_PATH", DEFAULT_OUTPUT_METADATA_REPORT_PATH),
        "max_llm_api_retry_attempts": max_llm_api_retry_attempts,
        "max_llm_validation_retry_attempts": max_llm_validation_retry_attempts,
        "run_log_file_path": os.getenv(
                    "RUN_LOG_FILE_PATH", DEFAULT_RUN_LOG_FILE_PATH),
    }

def remove_contradictory_unmapped_entries(
        llm_table_mapping: LlmSingleTableMapping,
        run_log_file_path: str,
        run_metadata_accumulator: dict) -> LlmSingleTableMapping:
    """
    Deterministically correct a specific, provable LLM self-contradiction:
    a field listed as unmapped while also appearing in field_mappings of
    the same response. The field_mappings entry is treated as the
    substantive claim; the unmapped entry is removed.

    Every removal is logged to the timestamped run log and recorded in the
    metadata accumulator under "auto_corrections". No other modification
    of LLM output is performed anywhere in this application.

    Args:
        llm_table_mapping: The LLM output, possibly self-contradictory.
        run_log_file_path: Path to the append-mode run log file.
        run_metadata_accumulator: Mutable run-report dict; removal records
            are appended to its "auto_corrections" list.

    Returns:
        The mapping with contradictory unmapped entries removed. (The
        input object is mutated and returned for call-site clarity.)
    """
    mapped_source_field_set = {
        entry.source_field for entry in llm_table_mapping.field_mappings}
    mapped_destination_field_set = {
        entry.destination_field for entry in llm_table_mapping.field_mappings}

    for list_attribute_name, mapped_field_set in (
            ("unmapped_source_fields", mapped_source_field_set),
            ("unmapped_destination_fields", mapped_destination_field_set)):
        original_field_list = getattr(llm_table_mapping, list_attribute_name)
        contradictory_fields = [
            field_name for field_name in original_field_list
            if field_name in mapped_field_set
        ]
        if contradictory_fields:
            correction_record = (
                f"Table '{llm_table_mapping.source_table}': removed "
                f"{contradictory_fields} from {list_attribute_name} because "
                "the same response maps them in field_mappings "
                "(LLM self-contradiction, deterministically corrected)."
            )
            append_timestamped_log_entry(
                run_log_file_path, "warning", correction_record)
            run_metadata_accumulator["auto_corrections"].append(
                correction_record)
            setattr(llm_table_mapping, list_attribute_name, [
                field_name for field_name in original_field_list
                if field_name not in mapped_field_set
            ])
    return llm_table_mapping


# ---------------------------------------------------------------------------
# Part 1: Deterministic schema-text parsing (no LLM involvement)
# ---------------------------------------------------------------------------

def read_schema_text_file(schema_file_path: str) -> str:
    """
    Read one schema .txt file and return its full content as a string.

    Args:
        schema_file_path: Path to the schema text file.

    Returns:
        Full file content.

    Raises:
        FileNotFoundError: If the file does not exist (with the path named).
        ValueError: If the file exists but is empty.
    """
    if not os.path.isfile(schema_file_path):
        raise FileNotFoundError(
            f"Schema file not found at path: '{schema_file_path}'. "
            "Check SQL_SCHEMA_FILE_PATH / MONGO_SCHEMA_FILE_PATH in .env."
        )
    with open(schema_file_path, "r", encoding="utf-8") as schema_file_handle:
        schema_text_content = schema_file_handle.read()
    if not schema_text_content.strip():
        raise ValueError(
            f"Schema file at '{schema_file_path}' is empty. "
            "It must contain the schema text from the assignment document."
        )
    return schema_text_content


def extract_brace_delimited_inner_content(
        full_text: str, opening_brace_index: int) -> str:
    """
    Given the index of an opening '{' in full_text, return the text between
    that brace and its matching closing '}' (exclusive of both braces).

    Character-level brace counting is used. This is safe for the assignment's
    pseudo-format because braces do not occur inside field names, types, or
    inline comments in this input.

    Args:
        full_text: The complete text to scan.
        opening_brace_index: Index of the '{' to match.

    Returns:
        Inner content between the matched braces.

    Raises:
        ValueError: If the character at opening_brace_index is not '{',
            or if no matching closing brace is found.
    """
    if full_text[opening_brace_index] != "{":
        raise ValueError(
            f"extract_brace_delimited_inner_content: character at index "
            f"{opening_brace_index} is '{full_text[opening_brace_index]}', "
            "expected '{'. This indicates a parsing logic error upstream."
        )
    brace_nesting_depth = 0
    for character_index in range(opening_brace_index, len(full_text)):
        current_character = full_text[character_index]
        if current_character == "{":
            brace_nesting_depth += 1
        elif current_character == "}":
            brace_nesting_depth -= 1
            if brace_nesting_depth == 0:
                return full_text[opening_brace_index + 1:character_index]
    raise ValueError(
        "extract_brace_delimited_inner_content: no matching closing brace "
        f"found for '{{' at index {opening_brace_index}. The schema text "
        "appears truncated or malformed."
    )


def extract_labeled_top_level_block(schema_text: str, block_label: str) -> str:
    """
    Locate a labeled block such as "tables": { ... } or "collections": { ... }
    in the schema text and return its inner content.

    Args:
        schema_text: The full schema file text.
        block_label: The label to search for, e.g. "tables" or "collections".

    Returns:
        Inner content of the labeled block (between its braces).

    Raises:
        ValueError: If the labeled block cannot be found.
    """
    label_pattern = re.compile(r'"' + re.escape(block_label) + r'"\s*:\s*\{')
    label_match = label_pattern.search(schema_text)
    if label_match is None:
        raise ValueError(
            f"Could not find a '\"{block_label}\": {{' block in the schema "
            "text. Confirm the input file matches the assignment format."
        )
    opening_brace_index = label_match.end() - 1  # index of the '{'
    return extract_brace_delimited_inner_content(schema_text, opening_brace_index)


def split_block_into_named_subblocks(block_inner_content: str) -> dict[str, str]:
    """
    Split a block's inner content into its top-level named sub-blocks.

    Used both for the SQL "tables" block (yielding table_name -> table body)
    and the Mongo "collections" block (yielding collection_name -> body).
    Nested sub-documents inside a body are NOT split here; they remain part
    of the body text and are handled by the dot-path extractor.

    Args:
        block_inner_content: Inner text of the "tables"/"collections" block.

    Returns:
        Dict mapping each top-level sub-block name to its inner body text.

    Raises:
        ValueError: If no named sub-blocks are found.
    """
    named_subblock_pattern = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*\{')
    subblock_name_to_body: dict[str, str] = {}
    scan_position = 0
    while True:
        subblock_match = named_subblock_pattern.search(
            block_inner_content, scan_position)
        if subblock_match is None:
            break
        subblock_name = subblock_match.group(1)
        opening_brace_index = subblock_match.end() - 1
        subblock_body = extract_brace_delimited_inner_content(
            block_inner_content, opening_brace_index)
        subblock_name_to_body[subblock_name] = subblock_body
        # Advance past this entire sub-block so nested '{' inside it
        # (Mongo sub-documents) are not mistaken for new top-level blocks.
        scan_position = opening_brace_index + len(subblock_body) + 2
    if not subblock_name_to_body:
        raise ValueError(
            "split_block_into_named_subblocks: found zero named sub-blocks. "
            "Confirm the input file matches the assignment format."
        )
    return subblock_name_to_body


def extract_flat_field_names_from_sql_table_body(sql_table_body: str) -> list[str]:
    """
    Extract the flat field names from one SQL table's body text.

    SQL table bodies in the assignment format have one field per line, e.g.:
        "emp_id":        INT            PRIMARY KEY

    Args:
        sql_table_body: Inner body text of one SQL table.

    Returns:
        List of field names in file order.

    Raises:
        ValueError: If zero field names are found (indicates parsing failure).
    """
    field_name_pattern = re.compile(r'^\s*"([A-Za-z0-9_]+)"\s*:', re.MULTILINE)
    extracted_field_names = field_name_pattern.findall(sql_table_body)
    if not extracted_field_names:
        raise ValueError(
            "extract_flat_field_names_from_sql_table_body: zero fields found "
            "in a table body. Confirm the SQL schema file format. "
            f"Body text begins: {sql_table_body[:120]!r}"
        )
    return extracted_field_names


def extract_dot_path_field_names_from_mongo_collection_body(
        mongo_collection_body: str) -> list[str]:
    """
    Extract flattened dot-path field names from one Mongo collection body.

    Nested sub-documents produce dot-paths, e.g. a "firstName" line inside
    a "fullName": { ... } block yields "fullName.firstName". Leaf fields at
    the top level yield their plain name, e.g. "_id".

    A line-based state machine is used:
        - '"name": {'  pushes 'name' onto the nesting stack.
        - '"name": <anything else>' records a leaf dot-path.
        - a line whose stripped form starts with '}' pops the stack.

    Args:
        mongo_collection_body: Inner body text of one Mongo collection.

    Returns:
        List of dot-path field names in file order.

    Raises:
        ValueError: If zero dot-paths are found, or if brace nesting is
            unbalanced (more closers than openers).
    """
    field_line_pattern = re.compile(r'^"([A-Za-z0-9_]+)"\s*:\s*(.*)$')
    nesting_name_stack: list[str] = []
    collected_dot_paths: list[str] = []

    for raw_line in mongo_collection_body.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue  # skip blank lines
        field_line_match = field_line_pattern.match(stripped_line)
        if field_line_match:
            field_name = field_line_match.group(1)
            value_remainder = field_line_match.group(2)
            if value_remainder.startswith("{"):
                # Opening a nested sub-document, e.g. "fullName": {
                nesting_name_stack.append(field_name)
            else:
                # Leaf field: record full dot-path from current nesting.
                collected_dot_paths.append(
                    ".".join(nesting_name_stack + [field_name]))
        elif stripped_line.startswith("}"):
            # Closing a nested sub-document.
            if not nesting_name_stack:
                raise ValueError(
                    "extract_dot_path_field_names_from_mongo_collection_body: "
                    "unbalanced braces (closing brace with empty nesting "
                    "stack). Confirm the Mongo schema file format. "
                    f"Offending line: {stripped_line!r}"
                )
            nesting_name_stack.pop()
        # Any other line content (e.g. stray commas) is intentionally ignored.

    if not collected_dot_paths:
        raise ValueError(
            "extract_dot_path_field_names_from_mongo_collection_body: zero "
            "dot-paths found in a collection body. Confirm the Mongo schema "
            f"file format. Body text begins: {mongo_collection_body[:120]!r}"
        )
    return collected_dot_paths


def extract_schema_label(schema_text: str) -> str:
    """
    Build the report label for a schema, e.g. "legacy_hrm (MySQL)".

    The database name comes from the "database" line and the short type
    (first word of the "type" value) from the "type" line.

    Args:
        schema_text: The full schema file text.

    Returns:
        Label string in the form "<database_name> (<short_type>)".

    Raises:
        ValueError: If the "database" or "type" line cannot be found.
    """
    database_name_match = re.search(
        r'"database"\s*:\s*"([^"]+)"', schema_text)
    database_type_match = re.search(
        r'"type"\s*:\s*"([^"]+)"', schema_text)
    if database_name_match is None or database_type_match is None:
        raise ValueError(
            "extract_schema_label: could not find both '\"database\":' and "
            "'\"type\":' lines in the schema text. Confirm the input file "
            "matches the assignment format."
        )
    database_name = database_name_match.group(1)
    # "MySQL (Relational)" -> "MySQL"; "MongoDB (Document)" -> "MongoDB".
    short_type = database_type_match.group(1).split()[0]
    return f"{database_name} ({short_type})"


# ---------------------------------------------------------------------------
# Part 2: LLM prompt construction, calling, and validation
# ---------------------------------------------------------------------------

def build_per_table_mapping_prompt(
        sql_table_name: str,
        sql_table_display_fragment: str,
        full_mongo_schema_text: str,
        sql_table_field_names: list[str],
        mongo_collection_dot_paths: dict[str, list[str]],
        previous_attempt_issue_messages: list[str]) -> str:
    """
    Build the user prompt for mapping one SQL table against the whole
    Mongo schema.

    Constraint compliance note: this prompt contains ONE SQL table fragment
    plus the ENTIRE Mongo schema. Both full schemas are never combined in
    a single prompt anywhere in this application.

    Args:
        sql_table_name: Name of the SQL table being mapped.
        sql_table_display_fragment: The table's text fragment, including
            its name and braces, for LLM readability.
        full_mongo_schema_text: The complete Mongo schema file text.
        sql_table_field_names: Deterministically-parsed field names of this
            table; the LLM must not invent names outside this list.
        mongo_collection_dot_paths: Deterministically-parsed dict of
            collection name -> list of valid destination dot-paths.
        previous_attempt_issue_messages: Validation issues from a prior
            attempt, included so the model can self-correct on retry.
            Empty list on the first attempt.

    Returns:
        The complete prompt string.
    """
    # Render the valid destination dot-paths per collection for the model.
    rendered_valid_destinations = "\n".join(
        f"- collection '{collection_name}': {', '.join(dot_paths)}"
        for collection_name, dot_paths in mongo_collection_dot_paths.items()
    )

    retry_feedback_section = ""
    if previous_attempt_issue_messages:
        rendered_issues = "\n".join(
            f"- {issue}" for issue in previous_attempt_issue_messages)
        retry_feedback_section = (
            "\n# Issues found in your previous attempt (fix ALL of these):\n"
            f"{rendered_issues}\n"
        )

    prompt_text = f"""# Database schema mapping task

Task: You are mapping one table from a MySQL schema to its semantically
equivalent collection in a MongoDB schema. You will see the one MySQL table
and the entire MongoDB Schema for context.

## Source for Map: one MySQL table
{sql_table_display_fragment}

## Destination for Map: complete MongoDB schema
{full_mongo_schema_text}

## Requirements:
1. Choose the single best-matching destination collection for this table.
2. Account for all source fields: each source field must appear either in
   field_mappings or in unmapped_source_fields. No field may be omitted.
3. source_field values must be exactly one of:
   {", ".join(sql_table_field_names)}
4. destination_field values must be a dot-path in the chosen
   collection (do not prefix the collection name). Valid dot-paths:
{rendered_valid_destinations}
5. type_transform: plain text, e.g. "TINYINT(1) -> Boolean"
6. confidence: a float between 0.0 and 1.0 (your estimation)
7. reasoning: one plain-English sentence, e.g. nature of map connection (e.g. exact match)
8. notes: value-transform logic if required (e.g. code lookups such as
   "A -> active"); can comment reason for high or low confidence, otherwise null
9. unmapped_destination_fields: dot-paths in the chosen collection that
   no source field of THIS table maps to
10. source_table must be exactly "{sql_table_name}"
{retry_feedback_section}
11. for primary-key -> _id mappings, notes must state the ID-generation strategy
Return only the structured mapping object."""
    return prompt_text


def call_mistral_for_single_table_mapping(
        mistral_client: Mistral,
        mistral_model_name: str,
        prompt_text: str,
        max_api_retry_attempts: int,
        run_metadata_accumulator: dict) -> LlmSingleTableMapping:
    """
    Call the Mistral API with structured (pydantic) output for one table,
    retrying on API/transport failure up to max_api_retry_attempts.

    Args:
        mistral_client: An initialized mistralai Mistral client.
        mistral_model_name: Model identifier, e.g. "mistral-small-latest".
        prompt_text: The complete per-table prompt.
        max_api_retry_attempts: Hard cap on API attempts before raising.
        run_metadata_accumulator: Mutable dict used for run reporting;
            this function appends per-call durations and error records to
            it (keys: "llm_call_durations_seconds", "errors").

    Returns:
        A validated LlmSingleTableMapping instance parsed by pydantic.

    Raises:
        RuntimeError: If all API attempts fail, or if the API returns a
            response with no parsed structured content.
    """
    most_recent_api_error: Exception | None = None

    for api_attempt_number in range(1, max_api_retry_attempts + 1):
        call_start_time = time.monotonic()
        try:
            # Structured-output call: the mistralai library converts the
            # pydantic model into a response schema and parses the reply.
            api_response = mistral_client.chat.parse(
                model=mistral_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a database schema-migration analyst. "
                            "You produce precise field mappings using only "
                            "the field names provided. You never invent "
                            "field names."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ],
                response_format=LlmSingleTableMapping,
                temperature=0.2,
            )
            call_duration_seconds = time.monotonic() - call_start_time
            run_metadata_accumulator["llm_call_durations_seconds"].append(
                call_duration_seconds)
            run_metadata_accumulator["llm_call_count"] += 1

            parsed_table_mapping = api_response.choices[0].message.parsed
            if parsed_table_mapping is None:
                raise RuntimeError(
                    "Mistral API returned a response but the parsed "
                    "structured content is None (structured-output parse "
                    "failure on the service side)."
                )
            return parsed_table_mapping

        except Exception as api_call_error:
            call_duration_seconds = time.monotonic() - call_start_time
            most_recent_api_error = api_call_error
            error_record = (
                f"API attempt {api_attempt_number}/{max_api_retry_attempts} "
                f"failed after {call_duration_seconds:.2f}s: "
                f"{type(api_call_error).__name__}: {api_call_error}"
            )
            print(f"[warning] {error_record}", file=sys.stderr)
            run_metadata_accumulator["errors"].append(error_record)
            # Simple linear backoff before the next attempt.
            time.sleep(min(2.0 * api_attempt_number, 15.0))

    raise RuntimeError(
        f"All {max_api_retry_attempts} Mistral API attempts failed. "
        f"Most recent error: {type(most_recent_api_error).__name__}: "
        f"{most_recent_api_error}"
    )


def validate_llm_table_mapping(
        llm_table_mapping: LlmSingleTableMapping,
        expected_sql_table_name: str,
        sql_table_field_names: list[str],
        mongo_collection_dot_paths: dict[str, list[str]]) -> list[str]:
    """
    Validate one LLM-produced table mapping against the deterministic
    field dictionaries. This is the anti-hallucination guard-rail.

    Checks performed:
        1. source_table matches the expected SQL table name.
        2. destination_collection exists in the Mongo schema.
        3. Every field_mappings.source_field exists in the SQL table.
        4. Every field_mappings.destination_field is a valid dot-path in
           the chosen destination collection.
        5. Every unmapped_source_fields entry exists in the SQL table.
        6. Every unmapped_destination_fields entry is a valid dot-path in
           the chosen destination collection.
        7. Coverage: every SQL field appears in field_mappings or in
           unmapped_source_fields (assignment requires full coverage).
        8. Every confidence value is within [0.0, 1.0].
        9. Not Both Mapped & Unmapped.
       10. Same disjointness rule on the source side.

    Args:
        llm_table_mapping: The LLM output to validate.
        expected_sql_table_name: The table this prompt was built for.
        sql_table_field_names: Deterministically-parsed field names.
        mongo_collection_dot_paths: collection name -> valid dot-paths.

    Returns:
        A list of human-readable issue messages. Empty list means the
        mapping passed all checks.
    """
    validation_issue_messages: list[str] = []
    sql_field_name_set = set(sql_table_field_names)

    # Check 1: table name integrity.
    if llm_table_mapping.source_table != expected_sql_table_name:
        validation_issue_messages.append(
            f"source_table is '{llm_table_mapping.source_table}' but must "
            f"be exactly '{expected_sql_table_name}'."
        )

    # Check 2: destination collection must exist.
    chosen_collection_name = llm_table_mapping.destination_collection
    if chosen_collection_name not in mongo_collection_dot_paths:
        validation_issue_messages.append(
            f"destination_collection '{chosen_collection_name}' does not "
            f"exist. Valid collections: "
            f"{sorted(mongo_collection_dot_paths.keys())}."
        )
        # Without a valid collection, dot-path checks cannot proceed.
        return validation_issue_messages

    valid_destination_dot_path_set = set(
        mongo_collection_dot_paths[chosen_collection_name])

    # Checks 3, 4, 8: per-field-mapping integrity.
    for field_mapping_entry in llm_table_mapping.field_mappings:
        if field_mapping_entry.source_field not in sql_field_name_set:
            validation_issue_messages.append(
                f"field_mappings contains source_field "
                f"'{field_mapping_entry.source_field}' which does not exist "
                f"in table '{expected_sql_table_name}'."
            )
        if field_mapping_entry.destination_field not in valid_destination_dot_path_set:
            validation_issue_messages.append(
                f"field_mappings contains destination_field "
                f"'{field_mapping_entry.destination_field}' which is not a "
                f"valid dot-path in collection '{chosen_collection_name}'."
            )
        if not (0.0 <= field_mapping_entry.confidence <= 1.0):
            validation_issue_messages.append(
                f"confidence {field_mapping_entry.confidence} for "
                f"source_field '{field_mapping_entry.source_field}' is "
                "outside [0.0, 1.0]."
            )

    # Check 8 (table level).
    if not (0.0 <= llm_table_mapping.confidence <= 1.0):
        validation_issue_messages.append(
            f"table-level confidence {llm_table_mapping.confidence} is "
            "outside [0.0, 1.0]."
        )

    # Check 5: unmapped source fields must exist.
    for unmapped_source_field in llm_table_mapping.unmapped_source_fields:
        if unmapped_source_field not in sql_field_name_set:
            validation_issue_messages.append(
                f"unmapped_source_fields contains '{unmapped_source_field}' "
                f"which does not exist in table '{expected_sql_table_name}'."
            )

    # Check 6: unmapped destination fields must exist in the collection.
    for unmapped_destination_field in llm_table_mapping.unmapped_destination_fields:
        if unmapped_destination_field not in valid_destination_dot_path_set:
            validation_issue_messages.append(
                f"unmapped_destination_fields contains "
                f"'{unmapped_destination_field}' which is not a valid "
                f"dot-path in collection '{chosen_collection_name}'."
            )

    # Check 7: full source-field coverage.
    covered_source_field_set = (
        {entry.source_field for entry in llm_table_mapping.field_mappings}
        | set(llm_table_mapping.unmapped_source_fields)
    )
    missing_source_fields = sql_field_name_set - covered_source_field_set
    if missing_source_fields:
        validation_issue_messages.append(
            f"These source fields of '{expected_sql_table_name}' appear in "
            f"neither field_mappings nor unmapped_source_fields: "
            f"{sorted(missing_source_fields)}. Every field must be covered."
        )

    # Check 9: a destination field must not be simultaneously mapped and
    # declared unmapped within the same table (internal contradiction).
    mapped_destination_field_set = {
        entry.destination_field
        for entry in llm_table_mapping.field_mappings
    }
    contradictory_destination_fields = (
        mapped_destination_field_set
        & set(llm_table_mapping.unmapped_destination_fields)
    )
    if contradictory_destination_fields:
        validation_issue_messages.append(
            f"These destination fields appear in BOTH field_mappings and "
            f"unmapped_destination_fields, which is contradictory: "
            f"{sorted(contradictory_destination_fields)}. "
            "Remove them from unmapped_destination_fields."
        )

    # Check 10: same disjointness rule on the source side.
    mapped_source_field_set = {
        entry.source_field
        for entry in llm_table_mapping.field_mappings
    }
    contradictory_source_fields = (
        mapped_source_field_set
        & set(llm_table_mapping.unmapped_source_fields)
    )
    if contradictory_source_fields:
        validation_issue_messages.append(
            f"These source fields appear in BOTH field_mappings and "
            f"unmapped_source_fields, which is contradictory: "
            f"{sorted(contradictory_source_fields)}. "
            "Remove them from unmapped_source_fields."
        )

    return validation_issue_messages


def obtain_validated_table_mapping(
        mistral_client: Mistral,
        application_configuration: dict,
        sql_table_name: str,
        sql_table_body: str,
        full_mongo_schema_text: str,
        sql_table_field_names: list[str],
        mongo_collection_dot_paths: dict[str, list[str]],
        run_metadata_accumulator: dict) -> LlmSingleTableMapping:
    """
    Run the call-then-validate loop for one SQL table until the mapping
    passes validation, or hard-error at the validation retry cap.

    On each retry, the validation issues from the previous attempt are
    included in the prompt so the model can self-correct.

    Args:
        mistral_client: An initialized mistralai Mistral client.
        application_configuration: Config dict from
            load_application_configuration().
        sql_table_name: Name of the SQL table being mapped.
        sql_table_body: Inner body text of the SQL table.
        full_mongo_schema_text: Complete Mongo schema file text.
        sql_table_field_names: Deterministically-parsed field names.
        mongo_collection_dot_paths: collection name -> valid dot-paths.
        run_metadata_accumulator: Mutable dict for run reporting; this
            function appends validation-issue records to it.

    Returns:
        A LlmSingleTableMapping that passed all validation checks.

    Raises:
        RuntimeError: If the validation retry cap is exhausted.
    """
    # Reconstruct a display fragment (name + braces) for LLM readability.
    sql_table_display_fragment = f'"{sql_table_name}": {{\n{sql_table_body}\n}}'
    max_validation_attempts = application_configuration[
        "max_llm_validation_retry_attempts"]
    previous_attempt_issue_messages: list[str] = []

    for validation_attempt_number in range(1, max_validation_attempts + 1):
        prompt_text = build_per_table_mapping_prompt(
            sql_table_name=sql_table_name,
            sql_table_display_fragment=sql_table_display_fragment,
            full_mongo_schema_text=full_mongo_schema_text,
            sql_table_field_names=sql_table_field_names,
            mongo_collection_dot_paths=mongo_collection_dot_paths,
            previous_attempt_issue_messages=previous_attempt_issue_messages,
        )
        llm_table_mapping = call_mistral_for_single_table_mapping(
            mistral_client=mistral_client,
            mistral_model_name=application_configuration["mistral_model_name"],
            prompt_text=prompt_text,
            max_api_retry_attempts=application_configuration[
                "max_llm_api_retry_attempts"],
            run_metadata_accumulator=run_metadata_accumulator,
        )
        llm_table_mapping = remove_contradictory_unmapped_entries(
                    llm_table_mapping=llm_table_mapping,
                    run_log_file_path=application_configuration["run_log_file_path"],
                    run_metadata_accumulator=run_metadata_accumulator,
                )
        validation_issue_messages = validate_llm_table_mapping(
            llm_table_mapping=llm_table_mapping,
            expected_sql_table_name=sql_table_name,
            sql_table_field_names=sql_table_field_names,
            mongo_collection_dot_paths=mongo_collection_dot_paths,
        )
        if not validation_issue_messages:
            print(
                f"[info] Table '{sql_table_name}': mapping passed validation "
                f"on attempt {validation_attempt_number}."
            )
            return llm_table_mapping

        # Log every issue clearly, then retry with feedback in the prompt.
        for issue_message in validation_issue_messages:
            issue_record = (
                f"Table '{sql_table_name}' validation attempt "
                f"{validation_attempt_number}/{max_validation_attempts}: "
                f"{issue_message}"
            )
            print(f"[warning] {issue_record}", file=sys.stderr)
            run_metadata_accumulator["validation_issues"].append(issue_record)
        previous_attempt_issue_messages = validation_issue_messages

    raise RuntimeError(
        f"Table '{sql_table_name}': LLM output failed validation on all "
        f"{max_validation_attempts} attempts. Final issues: "
        f"{previous_attempt_issue_messages}"
    )

def compute_globally_unmapped_destination_fields(
        validated_table_mappings: list[LlmSingleTableMapping],
        mongo_collection_dot_paths: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Compute destination dot-paths that no table's field_mappings target,
    across the entire run.

    These fields are not necessarily orphans: in this dataset they are
    typically embedded denormalized copies (e.g. employees.department.code)
    whose data originates from a source field already mapped to another
    collection (dept_info.dept_cd -> departments.code). The per-table,
    stateless LLM calls cannot express one-source-to-many-destinations
    mappings, so this deterministic aggregation check surfaces them for
    human review via the metadata report.

    Args:
        validated_table_mappings: All validated per-table mapping objects.
        mongo_collection_dot_paths: collection name -> all valid dot-paths.

    Returns:
        Dict of collection name -> sorted list of dot-paths never mapped
        by any table. Collections with full coverage are omitted.
    """
    globally_mapped_destinations: set[tuple[str, str]] = {
        (table_mapping.destination_collection, entry.destination_field)
        for table_mapping in validated_table_mappings
        for entry in table_mapping.field_mappings
    }
    unmapped_by_collection: dict[str, list[str]] = {}
    for collection_name, dot_paths in mongo_collection_dot_paths.items():
        never_mapped_dot_paths = sorted(
            dot_path for dot_path in dot_paths
            if (collection_name, dot_path) not in globally_mapped_destinations
        )
        if never_mapped_dot_paths:
            unmapped_by_collection[collection_name] = never_mapped_dot_paths
    return unmapped_by_collection

# ---------------------------------------------------------------------------
# Part 2 aggregation: cross-table collision detection
# ---------------------------------------------------------------------------

def detect_destination_field_collisions(
        validated_table_mappings: list[LlmSingleTableMapping]) -> list[dict]:
    """
    Detect potential collisions: cases where two or more source fields
    (possibly from different tables) map to the same destination
    collection + dot-path.

    Note on scope: because each LLM call is stateless and sees only one
    SQL table at a time (per the assignment constraint), collisions can
    only be detected here, deterministically, at aggregation time. This
    is documented in the write-up. Collisions are reported ONLY in the
    metadata report file, because the assignment output schema does not
    include a collision field.

    Args:
        validated_table_mappings: All validated per-table mapping objects.

    Returns:
        List of collision records, each a dict with keys
        "destination_collection", "destination_field", and
        "colliding_source_fields" (list of "table.field" strings).
    """
    destination_usage_registry: dict[tuple[str, str], list[str]] = {}
    for table_mapping in validated_table_mappings:
        for field_mapping_entry in table_mapping.field_mappings:
            destination_key = (
                table_mapping.destination_collection,
                field_mapping_entry.destination_field,
            )
            qualified_source_name = (
                f"{table_mapping.source_table}."
                f"{field_mapping_entry.source_field}"
            )
            destination_usage_registry.setdefault(
                destination_key, []).append(qualified_source_name)

    collision_records: list[dict] = []
    for (collection_name, destination_dot_path), source_field_list in \
            destination_usage_registry.items():
        if len(source_field_list) > 1:
            collision_records.append({
                "destination_collection": collection_name,
                "destination_field": destination_dot_path,
                "colliding_source_fields": source_field_list,
            })
    return collision_records


# ---------------------------------------------------------------------------
# Part 3: Final document assembly and output
# ---------------------------------------------------------------------------

def assemble_final_mapping_document(
        source_schema_label: str,
        destination_schema_label: str,
        validated_table_mappings: list[LlmSingleTableMapping]) -> dict:
    """
    Assemble the final mapping document in the assignment's exact format.

    Args:
        source_schema_label: e.g. "legacy_hrm (MySQL)".
        destination_schema_label: e.g. "people_platform (MongoDB)".
        validated_table_mappings: All validated per-table mapping objects,
            in source-schema table order.

    Returns:
        Dict matching the assignment output schema, ready for json.dump.
    """
    return {
        "mapping_version": MAPPING_VERSION_STRING,
        "source": source_schema_label,
        "destination": destination_schema_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # pydantic model_dump preserves the exact field structure defined
        # in LlmSingleTableMapping, which mirrors the assignment schema.
        "tables": [
            table_mapping.model_dump()
            for table_mapping in validated_table_mappings
        ],
    }


def write_json_file(output_file_path: str, json_serializable_content: dict) -> None:
    """
    Write a dict to disk as pretty-printed UTF-8 JSON.

    Args:
        output_file_path: Destination file path.
        json_serializable_content: The dict to serialize.

    Raises:
        OSError: Propagated with context if the file cannot be written.
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file_handle:
            json.dump(json_serializable_content, output_file_handle,
                      indent=2, ensure_ascii=False)
    except OSError as file_write_error:
        raise OSError(
            f"Failed to write JSON output to '{output_file_path}': "
            f"{file_write_error}"
        ) from file_write_error


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full pipeline: configuration, deterministic parsing, per-table
    LLM mapping loop with validation, collision detection, final JSON
    output (file + terminal print), and metadata report output.

    Any unhandled failure prints a full traceback and a specific error
    message, then exits with status 1. No errors are hidden.
    """
    pipeline_start_time = time.monotonic()

    # Mutable accumulator for the separate run-metadata report file.
    run_metadata_accumulator: dict = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "run_finished_at": None,
        "total_runtime_seconds": None,
        "model_used": None,
        "llm_call_count": 0,
        "llm_call_durations_seconds": [],
        "average_llm_call_seconds": None,
        "errors": [],
        "validation_issues": [],
        "destination_field_collisions": [],
        "auto_corrections": [],
        "globally_unmapped_destination_fields": {},
        "join_derived_field_proposals": [],
    }

    try:
        # ---- Configuration ------------------------------------------------
        application_configuration = load_application_configuration()
        run_metadata_accumulator["model_used"] = \
            application_configuration["mistral_model_name"]

        # ---- Part 1: deterministic parsing --------------------------------
        sql_schema_text = read_schema_text_file(
            application_configuration["sql_schema_file_path"])
        mongo_schema_text = read_schema_text_file(
            application_configuration["mongo_schema_file_path"])

        source_schema_label = extract_schema_label(sql_schema_text)
        destination_schema_label = extract_schema_label(mongo_schema_text)
        print(f"[info] Source schema: {source_schema_label}")
        print(f"[info] Destination schema: {destination_schema_label}")

        # 1.1: chop SQL schema into per-table bodies.
        sql_tables_block = extract_labeled_top_level_block(
            sql_schema_text, "tables")
        sql_table_name_to_body = split_block_into_named_subblocks(
            sql_tables_block)
        print(f"[info] Parsed SQL tables: {list(sql_table_name_to_body.keys())}")

        # 1.2: field-name dictionaries for both schemas.
        sql_table_field_dictionary: dict[str, list[str]] = {
            table_name: extract_flat_field_names_from_sql_table_body(table_body)
            for table_name, table_body in sql_table_name_to_body.items()
        }
        mongo_collections_block = extract_labeled_top_level_block(
            mongo_schema_text, "collections")
        mongo_collection_name_to_body = split_block_into_named_subblocks(
            mongo_collections_block)
        mongo_collection_dot_paths: dict[str, list[str]] = {
            collection_name:
                extract_dot_path_field_names_from_mongo_collection_body(
                    collection_body)
            for collection_name, collection_body
            in mongo_collection_name_to_body.items()
        }
        print(
            f"[info] Parsed Mongo collections: "
            f"{list(mongo_collection_dot_paths.keys())}"
        )
        for collection_name, dot_paths in mongo_collection_dot_paths.items():
            print(
                f"[info] Collection '{collection_name}': "
                f"{len(dot_paths)} dot-path fields."
            )

        # ---- Part 2: per-table LLM mapping loop ----------------------------
        mistral_client = Mistral(
            api_key=application_configuration["mistral_api_key"])

        validated_table_mappings: list[LlmSingleTableMapping] = []
        for sql_table_name, sql_table_body in sql_table_name_to_body.items():
            print(f"[info] Mapping table '{sql_table_name}' ...")
            validated_table_mapping = obtain_validated_table_mapping(
                mistral_client=mistral_client,
                application_configuration=application_configuration,
                sql_table_name=sql_table_name,
                sql_table_body=sql_table_body,
                full_mongo_schema_text=mongo_schema_text,
                sql_table_field_names=sql_table_field_dictionary[sql_table_name],
                mongo_collection_dot_paths=mongo_collection_dot_paths,
                run_metadata_accumulator=run_metadata_accumulator,
            )
            validated_table_mappings.append(validated_table_mapping)

        # ---- Part 2 aggregation: collision detection -----------------------
        # Collisions go to the metadata report ONLY: the assignment output
        # schema has no collision field, and per agreed scope all
        # unspecified data belongs in the separate report file.
        collision_records = detect_destination_field_collisions(
            validated_table_mappings)
        run_metadata_accumulator["destination_field_collisions"] = \
            collision_records
        if collision_records:
            print(
                f"[warning] {len(collision_records)} potential destination-"
                "field collision(s) detected; see the metadata report file.",
                file=sys.stderr,
            )

        # ---- globally_unmapped_destination_fields ------_______-------------
        run_metadata_accumulator["globally_unmapped_destination_fields"] = \
                    compute_globally_unmapped_destination_fields(
                        validated_table_mappings, mongo_collection_dot_paths)

        # ---- Join-derived (denormalized) field proposals -------------------
        # Deterministic stages: parse FK edges, find join anchors, attach
        # globally-unmapped embedded fields to their join-source tables.
        table_to_foreign_key_edges = parse_foreign_key_edges_from_sql_schema(
            sql_table_name_to_body)
        join_group_records = identify_join_groups_for_unmapped_fields(
            validated_table_mappings=validated_table_mappings,
            globally_unmapped_destination_fields=run_metadata_accumulator[
                "globally_unmapped_destination_fields"],
            table_to_foreign_key_edges=table_to_foreign_key_edges,
        )
        print(
            f"[info] Identified {len(join_group_records)} join group(s) "
            "for join-derived field proposal."
        )

        # LLM stage: one micro-call per join group, validated with retries.
        # Results go to the metadata report ONLY; mapping.json stays
        # strictly assignment-conformant.
        for join_group_record in join_group_records:
            join_source_table_name = join_group_record["join_source_table"]
            validated_join_derived_mapping_set = \
                obtain_validated_join_derived_mapping_set(
                    mistral_client=mistral_client,
                    application_configuration=application_configuration,
                    join_group_record=join_group_record,
                    join_source_table_body=sql_table_name_to_body[
                        join_source_table_name],
                    join_source_table_field_names=sql_table_field_dictionary[
                        join_source_table_name],
                    run_metadata_accumulator=run_metadata_accumulator,
                )
            run_metadata_accumulator["join_derived_field_proposals"].append({
                "destination_collection": join_group_record[
                    "destination_collection"],
                "embedded_parent_path": join_group_record[
                    "embedded_parent_path"],
                "join_source_table": join_source_table_name,
                "join_via": join_group_record["join_via"],
                "proposals": validated_join_derived_mapping_set.model_dump()[
                    "proposals"],
                "unresolvable_destination_fields":
                    validated_join_derived_mapping_set
                    .unresolvable_destination_fields,
            })

        # ---- Part 3: final output ------------------------------------------
        final_mapping_document = assemble_final_mapping_document(
            source_schema_label=source_schema_label,
            destination_schema_label=destination_schema_label,
            validated_table_mappings=validated_table_mappings,
        )
        write_json_file(
            application_configuration["output_mapping_json_path"],
            final_mapping_document,
        )
        print(
            f"[info] Mapping JSON written to "
            f"'{application_configuration['output_mapping_json_path']}'."
        )
        # Terminal print of the final mapping, per agreed scope.
        print(json.dumps(final_mapping_document, indent=2, ensure_ascii=False))

        # ---- Metadata report -----------------------------------------------
        pipeline_total_runtime_seconds = time.monotonic() - pipeline_start_time
        run_metadata_accumulator["run_finished_at"] = \
            datetime.now(timezone.utc).isoformat()
        run_metadata_accumulator["total_runtime_seconds"] = round(
            pipeline_total_runtime_seconds, 3)
        if run_metadata_accumulator["llm_call_durations_seconds"]:
            run_metadata_accumulator["average_llm_call_seconds"] = round(
                sum(run_metadata_accumulator["llm_call_durations_seconds"])
                / len(run_metadata_accumulator["llm_call_durations_seconds"]),
                3,
            )
        write_json_file(
            application_configuration["output_metadata_report_path"],
            run_metadata_accumulator,
        )
        print(
            f"[info] Run metadata report written to "
            f"'{application_configuration['output_metadata_report_path']}'."
        )
        print(
            f"[info] Pipeline completed in "
            f"{pipeline_total_runtime_seconds:.2f}s with "
            f"{run_metadata_accumulator['llm_call_count']} LLM call(s)."
        )

    except Exception as pipeline_error:
        # Full traceback plus a specific message; no error hiding.
        traceback.print_exc()
        print(
            f"[fatal] Pipeline failed: {type(pipeline_error).__name__}: "
            f"{pipeline_error}",
            file=sys.stderr,
        )
        # Best-effort: still attempt to persist the metadata report so the
        # failure context (errors, call counts, timings) is not lost.
        try:
            run_metadata_accumulator["run_finished_at"] = \
                datetime.now(timezone.utc).isoformat()
            run_metadata_accumulator["total_runtime_seconds"] = round(
                time.monotonic() - pipeline_start_time, 3)
            run_metadata_accumulator["errors"].append(
                f"FATAL: {type(pipeline_error).__name__}: {pipeline_error}"
            )
            write_json_file(
                os.getenv("OUTPUT_METADATA_REPORT_PATH",
                          DEFAULT_OUTPUT_METADATA_REPORT_PATH),
                run_metadata_accumulator,
            )
            print(
                "[info] Partial metadata report written despite failure.",
                file=sys.stderr,
            )
        except Exception as metadata_write_error:
            # Report-writing failure must not mask the original error.
            print(
                f"[warning] Could not write partial metadata report: "
                f"{type(metadata_write_error).__name__}: "
                f"{metadata_write_error}",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()

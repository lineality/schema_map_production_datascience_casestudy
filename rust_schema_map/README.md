DB-Mapping Fragment Accumulator:  Rust
Geoffrey Gordon Ashbrook 2026.07.09-10th

(note: See full assignment instructions in doc.)


# Overall Plan:
- Use Rust
- input two db~schema files
- use rules-based NLP to map fields
- Produce a json file of the map


A .py LLM version (with dubious practical applications) exists, but can a production Rust version be made? Let's explore.

### Rust & Levinshtein Distance et al
This task is a schema-matching problem, which is much more structure-intensive than many very-unstructured-input uses where a generative LLM may be needed.  Schema-matching has decades of history and literature for GOFAI/statistical techniques (Levenshtein-distance, token-set similarity, structural/type compatibility, abbreviation dictionaries, and many more) that are more production-friendly than LLMs of the 2020s.

(E.g. see https://datamade.us/blog/schema-matching/ )

A Rust implementation, namely a vanilla-rust (standard-library-only) approach, may have significant advantages over python:
- versatility of deployability
- slimmer: efficiency
- faster
- safer
- more robust atomics, concurrency, parallelization
- better error tracking and auditing
- explainability
- predictability
- maintainability
- memory-safe (various benefits possibly including strict handling of secure values)
- long term asset: python LLM code is highly unlikely to still run in 50 years, and probably not even 5 years (possibly not even one year), whereas this Rust code would be a long term asset, likely to run indefinitely.
- no third party dependencies
- strict production-grade mode and case/error handling
- more meaningful confidence-score

Note: It took significantly longer to design and make the Rube-Goldberg LLM-fake solution-in-search-of-a-problem toy, compared with a Rust-NLP-GOFAI system that does have the potential to be useful in production and general practical use.

Note: Code built with support from Anthropic Claude, using specific mode, case-handling, and standards rules.

# NLP Approach Summary: Old Fashioned NLP Approach Overview
## Schema Field Mapper — Design Overview (Rust, deterministic, no LLM)
The overall project-context is moving records from an old database to a new database.

Two databases have the same employee record information, but they are structured differently. One is an old SQL database with short, cryptic column names such as `emp_cd` and `dept_id`. The other is a newer document database (MongoDB) with descriptive, nested names such as `employeeCode` and `department.departmentId`.

To move data from the old to the new, all fields in the old schema needs to be matched to their equivalent fields in the new schema. Where the names or formats do not clearly line up, there needs to be a record of how a match was made and what needs to change about the data itself (e.g. converting 0/1 to true/false).

This case (not too uncommon in real life, needing to compare two databases) is an interesting case straddling the line between computer-code and human-unstructured language. In some cases two databases may be highly similar and regular, and the best way to map them would be to use a rigorous program that cannot make any 'fat finger' mistakes. In other cases the two databases may only be understood (mapped) by talking with people who understand the fields and their uses (where domain-knowledge needed for interpretation).

In this case, we can likely make a good rules-based system. But the input-source-documents about two schemas are human-notes in a hodgepodge of formats, not in a real machine format that can be simply read into a program.

A generative AI model can do, or help with, this task, but the GenAI-model may make either very human-like or not-so-human-like mistakes (such as inventing entirely new hallucinated fields (which...even moderately large models still do on this task in 2026)). Also, using an LLM 'in production' is expensive, unreliable, fragile, etc.

This Rust-NLP approach aims to use plain, rule-based code to show that this kind of matching can be done reliably, quickly, and in a way that is fully explainable and suitable for 'production' software in real world use.

This approach breaks the problem into five parts or steps:


## Five-Step Process

#### 1. Read and Process both schemas:
The program reads each schema text file and extracts information about every field:
- which table/collection it belongs to,
- its name,
- its data type,
- any special markers (like "this is a unique ID" or "this points to another table"), and
- any explanatory comments written in the notes.

#### 2. Match up the tables (before trying to match each field):
(Note: "naming things is hard" and SQL and Mongo use different jargon-terms for the same thing: SQL tables are called "tables", whereas Mongo tables are called "collections")

Before attempting to match individual fields, the program figures out which SQL-table in the old schema corresponds to which mongo-collection in the new one (e.g. `emp_master` to `employees`) by comparing their names.

#### 3. Compare All Fields and Score All Comparisons:
For each table pairing, every old field is compared against every new field, and each pair gets a ~same-ness or "similarity" or "distance" score.

For a human or an LLM, 'same-ness' is based on an intuitive recognition (what Daniel Kahneman and Amos Tversky called a "System 1" process that is faster than deliberate logic-calculations ("System-2", which is slow and arduous for people, and for Gen-AI-LLMs).

But this type of use-case is probably a good candidate for using clear metrics for evidence-based same-ness (for example, measures/evidence that can be peer-reviewed, audited, error-checked, and traced-back to something concrete, even if the score was not optimal). The json-report fields of 'reason,' 'confidence,' and 'notes' should be something of a red-flag: for a human or Gen-AI these text-fields are extremely noisy and arbitrary, whereas what would be really valuable to have here is data that are clearly defined and ideally that can be used directly by software not only for a committee of debating people (something more like a Rust Struct+enum enforced by the rust-compiler than a reddit-post).

Our Rust-NLP same-ness score comes from combining several kinds of evidence (using some time-tested old-fashioned techniques):

> - Do the words in the field's own name match? Each field name (e.g. dept_cd, or code) is broken into its component parts, abbreviations are expanded using a built-in dictionary (that can be added to over time), and the sets of parts of the two database-fields are compared for overlap. We can also make rules to give some parts higher or lower value. If a part of the field's name is also in the name of the parent table that it lives in the it has a lower value because it does not help tell one field apart from its neighbors in the same table. E.g. in the "department" table, the part "dept" or "department" appearing all or many department fields does not help us distinguish which department-field is which, so that part of the name ("dept") counts for less in a match. You could think of this as flagging that signal as noisy, and, importantly, doing so for an auditable reason.

> - String-Distance: Are the names similar as raw text? While people can intuitively see two text-strings as similar, there are also quick, reliable, rigorous ways of measuring similarity or 'distance.' Two time-tested ways used here are levenshtein distance and jaccard-similarity
(https://en.wikipedia.org/wiki/Levenshtein_distance, https://www.ibm.com/think/topics/jaccard-similarity).

> - Do the human-written annotation-comments match? If the comment-annotations for both fields share references (like a reference to "ISO 4217" (for three-letter alpha-numeric codes), that counts as evidence.

> - Do the data types make sense together (or are they identical)? A clear example of this is two fields both having a "date" data-type. That kind of match signals another small vote for same-ness.

> - Are there structural clues (e.g. primary-key unique-identifiers, or pointers to other tables)? Two common 'structural' features of databases (even different types of databases) are 1. unique-id fields and 2. references to other tables. Sharing a 'unique-id' status is simple for our rules: if a source column is marked as the table's unique-id field, and the destination field is mongoDB unique id, then that is a big vote of confidence that these are the unique-id (or 'primary key') fields for these tables (there is only one unique-id field per table). The other example area here, 'cross-references' between tables, is a great example of how a low-score signal can be very valuable. Some columns do not identify their own row, they are references that point to a row in a different table. For example, in the old schema, dept_info.dept_head_id is marked FK -> emp_master.emp_id, meaning that this column holds the ID of a row in the employee table (a reference pointing to the department's head employee). Since we already matched the tables name emp_master and employees (back in step 2), we can programatically verify if a source field's reference target and a destination field's reference target refer to tables that are paired with each other, or not! It is valuable to check if the references match. For example, the dept_info table also has a parent_dept_id, marked as FK -> dept_info.dept_id (a department pointing to its parent department), not to an employee. And the new schema's departments collection has both parentDepartmentId (pointing to departments) and headEmployeeId (pointing to employees). The names alone can look deceptively similar, but their reference targets do not match! One points at employees, the other points at departments. When our program sees a source field and a destination field both declaring references, but they are pointing at tables/collections that are not ones already paired together, we can count this as evidence against the match: an active penalty, because two reference fields pointing at different real-world things are unlikely to be the same field, no matter how similar their names look. In a real-life database with tens, hundreds, thousands, or more, of these references, having a reliable, auditable, fast, systematic way of scrutinizing this web is much better than relying on intuition. Programmatic approaches scale well.

> - Detecting Boolean (true/false) Options: Once we detect what fields are boolean, we can more easily compare them. Many fields may be effectively-boolean but not look like it at first. A field that uses only two coded values (like "A=Active, I=Inactive") can be recognized as equivalent to a boolean (true/false) e.g. Active=True/False vs. "A=Active, I=Inactive".

#### 4. "Greedy" Algorithm, use the Weight: Pick best matches & do not force bad matches: Up to this point we have done analysis of tables and collected measurable evidence as "weights" for and against the 'same-ness' of each possible matching of an old-db-field to a new-db-field. The next step is: Let's make our decisions on the matching. Let's use these weights to see what the best match is for each of the old fields. We will use what is sometimes called a 'greedy' algorithm. Wikipedia puts it as: "A greedy algorithm is an algorithm which, at each step, makes the choice that is locally optimal, and subsequently does not reconsider past choices." We move through in one pass, not changing a past decision based on something we might find later on. For each possible field-pair, the program picks (from the not-yet-matched-to new fields) whichever (available) new field scored highest, if (only if) that score clears a minimum confidence bar. If nothing scores highly enough (for example, a field like `dob`/date-of-birth genuinely has no counterpart in the new schema), then it is left unmatched instead of being given a bad 'best-match.'


**5. Write out the results.**
The program produces a JSON report listing every match, its confidence score, a plain-English explanation of why it was matched, and (where relevant) instructions for how the data need to be transformed (e.g. "value 'A' means Active, which is now a clear boolean and so becomes the value `true`").

A match's reported confidence is lowered when a runner-up candidate scored nearly as well. When this happens, the match's explanation says so.


## Background Data for Audits and Reviews:

The program also writes a supplementary report showing the *top three* candidate matches considered for each field, including their scores. A reviewer can see exactly how close a decision was. For instance, in this run two fields were matched by a very thin margin, and the report shows how the winning match beat the runner-up. This type of transparent 'paper trail' is something the Generative-AI-based approaches typically (for example when not combined with a systematic approach like this, or even an embedding-vector distance measure) cannot provide with meaningful detail and consistency.


## Engineering choices, explained

- **No external libraries.** Everything — including reading files and writing JSON — is written from scratch, so the program has no hidden dependencies. This ensures that future-builds will not be prevented by a dependency changing. This prevents third-party security issues. This also helps significantly for deployment; this code can run on devices and within environments where a python dependency-spaghetti-monster would be impossible or infeasible.

- **Production case-handling to not crash.** Both for Rust-Rules and for a specific Mode & Code-Handling policy, the paradigm of this design aims to be effectively within NASA's Power-of-10 rules for system-critical software, as updated for systems-programming in 2026. The strictness of the rules plus Rust's narrow compilation rules should help to ensure that an expected range or failure will be handled gracefully, rather than a crash that in production may be a security or privacy liability. Python will perhaps never be able to fully separate debug-mode from a production-build, but Rust is oriented towards a clearly separate (and separately testable) production-release code compilation. Error handling and cargo tests can still be added-to over time, to be even more ridiculously robust, but 'out of the box' this is leagues beyond what python is able to offer. This is a very-rapid-prototype, for a nonsensical non-production academic-study task, but it represents an approach that can be productionized for a production-use-case (unlike python which is inherently not suited for many real world production cases).



## Not infinitely flexible automatically:

Some of the resources for matching here are per-dataset rules based on this specific use-case and set of schemas. This is different from an LLM approach and each has pros and cons (which likely suggest ways to combine approaches).

And LLM is more flexible in some ways, it can handle much more unstructured language. But 'pretrained' models ("it's in the name"), contrary to popular misunderstanding, are not made to learn or be updated. An LLM is likely to be able to handle more tasks, but for tasks it cannot handle, that's the end of the road (then it's time to spend a few billion more dollars training the next disposable model).

As with the type of rules-based tool here, if you use significantly different input schemas (such as retail inventory data), it is possible that the data would be clean enough to work, but it would be likely that new 'domain knowledge' and edge-case handling would need to be added to this tool.

This could be seen as a limitation, but this could also be seen as a powerful feature: This tool CAN be updated, more and more indefinitely.

1. Maintainability in General: This is not a fragile black box doomed to become an unmaintainable artifact of history. This software is designed to be friendly to future updates and improvements.

2. Accessibility: A rust program like this is not only theoretically maintainable, it is realistically maintainable. Any person (especially with help from an LLM) can add rules or any modules to this modular rules-based system.

3. The audit report is designed to make updating evidence-based: reports show exactly which fields scored too low to match, so a person knows precisely where to add domain knowledge.

This program could also easily be adapted into being a module to add to other systems, or into another architecture (such as an endpoint, or serverless endpoint, or part of some other pipeline).


## Matching and Joint-Derived Fields:
Separately, a few new-schema fields (like the full department or location details attached to each employee) are deliberately left unmatched at the employee level — not because the matcher failed, but because those data do not come from the employee table at all;. Fields derived from joins between tables are, by standard practice, not included in this level of field-mapping but in a process call "normalization" (yet another colliding term in STEM, naming things is hard (apparently)) those derived fields handled later by combining data from the department and location tables during the actual migration.

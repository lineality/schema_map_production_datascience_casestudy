DB-Mapping Fragment Accumulator
Geoffrey Gordon Ashbrook 2026.07.09th

Rule From Instructions: 
"You cannot pass both schemas to an LLM in a single prompt and receive a finished mapping."

(See full assignment instructions in doc.)


# Overall Plan:
- python application
- input two db~schema files
- Produce a schema to schema map json (definition issues aside)
- supplementary reports: metadata + join-derived fields, error-log
- standard checks and guard-rails against erratic behavior (e.g. field hallucination protection, conflicting results, etc.)


#### In this case, for MVP-1: MySQL-table -> Mongo-collection
A design question is how general and flexible to make this, e.g.
- formats of input
- diversity of types of databases

Note: the lack of an overall consistent set of terms (e.g. each DB being able to significantly redefine or omit terms such as 'schema' and 'table') makes a clear description of an overall flexible 'any-DB' framework problematic.

Here the terms 'table' and 'collection' may be hard-coded, but that may not be suitable for a real-world production application.

For MVP-1, the input will be .txt prints of the schemas, and there will only be two known types. But this hard-coding is likely not realistic for a more flexible production system (probably).


#### Designed to be modular ~scalable, including: 
A. Being able to iterate through large schemas one table/collection at a time
B. Being able to use smaller-models with smaller-inputs.

This activity has aspects of an arbitrary dummy exercise, but can be made and discussed to elucidate real-world factors.

The plan here will be to chop one schema into small-enough fragments (one table or collection at a time), and compare that to the entire other schema (see discussion below about contextual analysis and collision prevention).

Whether this is a hard-coded tool to only be used once ever, or a semi-hard-coded tool to be used for a few related projects, or on the other end of the spectrum is meant to be a fully generalizable production-deployed tool, greatly impacts the design decisions. For example, if this is used only once or a few times for a specific case, other features that are helpful may be a high priority (but which would be impossible to include for a general any-db production system), such as hand-holding for how to handle (perhaps known) ambiguous, colliding, and orphan fields. Or having redundancy features to make use of large-context inputs and reduce and check hallucination and inconsistencies vs. trying to design an on-edge optimized micro-tool. 

As is this design-spec seems to be half Model-T-Ford, half stealth-bomber, and doubtfully useful.





#### Real World Factors
The design details of this particular abstract-activity are sparse and dubious, but in specific real-world situations there are ~parameters that will guide design choices, such as: 
- range of input scale
- range of available model size
- range of available model type
- exactly how and where the model is deployed (e.g. on edge, in-house-cloud, etc.)
- cost limit requirements (e.g. cannot use frontier models, or not frequently)
- speed limit requirements (e.g. must be sub-second; or less than 30 sec)
etc. 

Since this system is generating unstructured text presumably for a human to read, and where the scale is within 2026 token-length context-size limits, and given the significant scope-expansion of an 'increment-crawl' architecture, it is notably unclear why this project is designed as an incremental unstructured-blob generator with small input size. The 'scale' rules are also strangely arbitrary: you can input one whole schema, but not both. That makes no sense. 

That said, in the real world there are cases (if not like this 'artificial-artifact' case) where input needs to be processed into modular parts, for example such that neither "schema" could be input all at once, and where the production model size is significantly small (e.g. less than 1B, and then further quantized) limiting the scope of each prompt and the scope (e.g. field quantity) of the output.

### SQL Table Extraction
Further to this problematic ambiguity, for this one-off case it likely makes the most sense to use a deterministic python parsing of the SQL schema, but this makes the system much less general. This is an area where design needs to match how the application will really be used, and this case shows how ambiguity (which can easily arise from bad project management and chaotic planning) likely lead to future problems.


#### Clear Python
To strictly follow python best practice including:
- types
- error handling
- try-except
- very clear error messages
- meaningful unique not-colliding names
- thorough doc-strings for future devs
- clear line comments
- valuing communication and maintainability


#### Framework (if not in MVP-1) will allow the user to configure what API to use (if only in flexibility of design):
- Mistral (MVP-1)
- Anthropic
- Google


Note: input structuring (maybe pydantic...)

Note, while some will rush to consider pydantic output structuring using a cloud-api service to be shiny best practice, it is important to understand that this reliance on high level third party dependencies (while it may have short term advantages, as third party add-ons often claim) does not simply translate to working directly with self-hosted (or in-house-trained and built) models. In some cases cloud-service 'output format' options may be best, but the tradeoffs and risks of black-box hidden 'just make it go away' solutions should be noted.

(if not for MVP-1): Consistent with disallowing whole-schema input is using smaller more efficient models. For production to run with the smallest-model possible, either the size and form of pydantic structure inputs may be too big, too elaborate, the cloud-service feature of 'output format' may not be available, third party libraries may not be (feasibly) available where the pipeline is deployed (e.g. inside an AWS-lambda-function) or some other issue may prevent approaches and tools such as pydantic output formatting. Another example may be in comparing model performance, you may want to plug in any model from hugging-face to see how it does, but making your pipeline only for a cloud-service-api can introduce barriers to a timely comparison (e.g. even if you host XYZ model in your cloud and rig up an api, it will not have magic input and output abstractions (because those are not part of models, those are part of high level hidden-service-bundles); not understanding this would be bad.





## Part 1. Setup:

1.1 Break SQL schemas into per-table modules
- produce a python list of strings for the sql-tables:
- for MVP-1, use python rules on an input .txt file to chop the schema into separate tables, so those tables can be (according to the extremely arbitrary instructions) fed into the main process.

(alternately, the tables could be hand-divided and hard coded into the application as a list of text blocks. For a one-time-only application, directly putting in the input without intermediate processing may be best.)

sql_table_list = []

- This will also be used to confirm that keys match (e.g. that later stated tables and fields can be traced back to exist here)


1.2 table-field-name dict
if possible by python rules (if not, by llm) make a dict of field-names per table for both SQL and Mongo.

Store full flattened dot-paths (fullName.firstName, employment.startDate, ...) and use that to validate LLM-returned destination_field.

(Again, this could just be hard-coded to bootstrap. This is another artificial area, most likely in real life the user would have access to a digital list of table-field names, not a paper sheet.)

#### 1.1 vs. 1.2
- 1.1 is chopping up the unstructured text doc describing the SQL schema into tables: only for SQL, no need to chop up the Mongo "schema"
- 1.2 is making a field-name data structure for both SQL and Mongo


1.3 Make aggregation data structures
like this, roughly: see assignment doc for exact details
```
  "mapping_version": "1.0", 
  "source": "", # extract from doc
  "destination": "", # extract from doc
  "generated_at": "", # timestamp datetime fields
  "tables": [{}]
```

1.4 Other Data

extract table data for the final json map report & other use
e.g. for 
- get timestamp for  "generated_at": "<ISO 8601 timestamp>",
- "source": "legacy_hrm (MySQL)",
- "destination": "people_platform (MongoDB)",
- something for collision-data? e.g. details of potential collisions detected)


## Part 2. Per Table/Collection to Schema Compare & Aggregate

2.1 The LLM suggests the field_mappings data (all relevant fields)
- Iterate through each sql-table and compare it to the whole mongo ~'schema' to find potentially matching fields.
- Use generative LLM to produce these table-field-to-collection-field comparisons.
> LLM-api should return a complete per-table mapping object — `destination_collection`, table-level `confidence` and `reasoning`, and `field_mappings` per SQL table

Note: one area where there can be error checking and deterministic error correction is if a mapped field appears in an "unmapped_*_fields" list. This can be both error-logged, and manually corrected (removed from the unmapped-list).

2.2 - Produce results dicts

2.3 Aggregate results dicts:
- Check for potential collisions
- Check for non-existent fields
- etc.

```
{
  "mapping_version": "1.0",
  "source": "legacy_hrm (MySQL)",
  "destination": "people_platform (MongoDB)",
  "generated_at": "<ISO 8601 timestamp>",
  "tables": [
...             ],
      "unmapped_source_fields": [],
      "unmapped_destination_fields": []
    }
...
]
}
```

Loop (all of part 2)


Note:
The assignment doc is the only source on this fake input data.
I can copy and paste those schemas into files or to hard code them as python strings. The fake assignment is the only source of the fake data. 

This is obviously not helpful if the scope is designing a general real world production application with unspecified input and input format.

This is also irregular as the 'format' of the fake data appears to be a hand-written pigeon of different formats (not valid SQL DDL, not valid JSON (unquoted types and no commas)), not something that would be machine-output by any real system. In production these points would be relevant because it is important whether or not the input schema-naming is authoritative and strict, or if it is an approximate sketch on a blurry napkin. Both could be real-world cases, to be fit by different solution designs.

- for LLM api: basic retry N times (e.g. 10) before hard error



## Part 3. Produce Final Json
- any final checks?
- output map_json: save file, terminal print,


### Output Meta-Data: 
Any data not-specified should in the original assignment spec should be reported in a separate report file:

- date time 
- errors, 
- issues, 
- time,
- how many LLM calls
- average time of LLM call 
- which model used
- etc,

(ideally, total tokens and avg tokens, not MVP-1 (each API differs))

#### Errors Etc.
If the LLM output contains an incorrect field, log that (clearly) and try again (until there are no fake-fields) (hard error at N tries (e.g. 10))



# Discussion

#### Collisions and Statefulness:
To me a significant part of this project should be pointing out potential "collisions", e.g. this tool seems to be generating documentation to help people to understand what is clear and what is unclear when attempting to map two different (and different types of) database ~schemas together. (Note: Mongo uses a different definition of 'schema,' but we have to describe this somehow.) 

Yet, by preventing whole schemas from being put into the same input 'context,' and by omitting collisions from the json report, this may walk straight into the worst aspect of amnesiac stateless 'agentic' automation, where the json report could generate an endless exhaustive list of possible matches (made without (by design, for some reason) ever directly comparing the two schemas.

The entire point of using an LLM, over a GOFAI-NLP line-by-line system, is that the LLM can look at both schemas in one context. Contextual interpretation is this super-power of foundation models. By segmenting the inputs into chucks (by mysterious design) you remove that context.

Trying to reverse-engineer a collision-detection system after you deliberately generated decontextualized guesstimations is a self-imposed challenge that, depending on how it is defined, may not be possible to fix. For example, if for some reason you needed data from more than two records to see that there is an issue, incrementally showing just two records (one from each schema) to an LLM any number of times and in any number of combinations will never include that larger context, because the LLM is stateless (which many people do not understand). 

By chopping the process up, there is at least a task (if not an impossible task) of re-integrating isolated separate guesstimations that (for whatever reason) are not allowed to be compared to the original whole schema).

For example, given an especially vaguely named set of schemas (e.g. where there are half a dozen items all called "key" and "data" and "id" (which I very often see indefinitely confusing people in meetings in real life)), it is very likely that the overall json-report will contain many colliding matches with arbitrary confidence scores. What is likely to happen when a person is given such a json-guide? The outcome will be either minorly or majorly bad. 

As another type of collision or blindspot case:
dept_info.dept_cd feeds both departments.code and employees.department.code. Due to the self-imposed blindness-to-context, stateless per-table calls structurally cannot see or show one source field feeding two collections.

Also, fields that would be the result of a join, such as SQL's employee department, can't be obtained without statefully understanding the scheme structure (which for, some reason was, intentionally prohibited). 


MVP-1 uses pydantic 

### Join-derived-fields

The assignment-definition does not mention join-derived fields (such as employee-department. The strictly-defined output format cannot express join-derived-fields. A tables[] entry contains one destination_collection, and each destination_field is a dot-path within that collection. 

A join-derived mapping is, e.g.
```
dept_info.dept_nm -> employees.department.name 
```
like a dept_info source field landing in the employees collection. The dept_info table object (whose collection is departments) has no way to express that.



### Goals: Three Versions


A: .py Version
- start with input of two files (can be hard-coded into .py main(), or the .env)
- using py dot-env (and a .env file)
- venv env
- standard ```
if __name__ == "__main__":
main()
```
- if file paths not in .env, Q&A input() ask the user.
- user creates .env including which model


B: Colab Version

C: Rust & Levinshtein Distance et al
This task is a schema-matching problem, which is much more structure-intensive than many very-unstructured-input uses where an LLM may be needed.  Schema-matching has decades of history and literature for GOFAI/statistical techniques (Levenshtein, token-set similarity, structural/type compatibility, abbreviation dictionaries) that are more production-friendly than LLMs. 

https://datamade.us/blog/schema-matching/ 



### Notes from Testing:
- Mistral Small (while not a tiny model, is not able to follow the task)
- Mistral Medium is able to do a decent job

### Production Contexts:

Schema Mapping Production-Data-Science Case Study 
Production Data Science & Software Case-Study Notes

Interacting Structured and Unstructured data, with Generative, or non-generative output (Embedding-Vector, classification, regression, etc.) 

This is an excellent example for looking at how and where production data science projects need to be managed carefully.

## Connected Areas:
- needs and goals evaluation https://github.com/lineality/needs_goals_assessment_disambiguation 
- project areas https://github.com/lineality/project_areas_for_project_and_product_management 
- coordinated decisions https://github.com/lineality/Networked_Voting_and_Decisions_Including_One_Time_Pads 
- project-definition: definition behavior studies https://github.com/lineality/definition_behavior_studies 


## Production-Design Topics:

1. local, batch, at-scale, cloud,

2. Output structuring and details of deployment
- self-hosted, 
- .gguf,
- hidden api-service
- specific api-services

3. Classic NLP, GOFAI, "Deterministic" approach, 

4. Crawling vs. large-chunk

5. Paralleliszation
- also for debugging

6. Language Choice
- third party dependencies

7. Third Party dependencies
- a serious and escalating liability
- See the 'where can be used' issue for pydantic + output_format structuring.

8. Full automation vs. semi-automating tool
- the fact that this dummy-project generates an unstructured text-field is suspicious, suggesting that this a solution in search of a problem, automatically and verbosely generating possibly useless and illogical documentation-word salad that some human being will manually need to inspect, as opposed to a tool designed to be used by a person for something more specific.

Small-Clear-Task-Doer: Good, Best
Big-automated-task-doer: Dubious, but can be good.
Task-Helper: Good, flexible.
Automated-Unchecked-Documentation-Generator: Very Bad. 

9. Definition, Testing, Evaluation, Benchmarking, Auditing:
Is something designed so that what it is doing is clearly defined in such as way that it can, effectively, be 
- unit-tested
- workflow-tested
- performance benchmarked (as in live month-to-month performance with potential changes to the system or to inputs)
- given a meaningful evaluation test (not a mismatched test, or a dubious or overly-indirect test)

10. Error logging and error/case handling 
- also gets into language choice

11. Atomics, Parallelism and Concurrency
- can it be optimized, does need to be
- debugging
- maintainability
- the project-scope and time required
- Language Rust vs. Python

12. scalability

13. maintainability

14. deployability
- on edge
- in a serverless endpoint
- consistently fast enough to be an endpoint under 30-sec. (backend-frontend latency, the need for 'step functions' etc.)

15. Design Maturity
Is the whole project/product design mature or is this a theoretical 'solution in search of a problem,' with hopes of a deployment and affordability pathway, and with a dream of maintainability? 

16. A case for a "Small" vs. "Large" Foundation model

17. Input data type ambiguity:
For document processing in real life this can be the most critical issue, yet it can be overlooked all the way until eventual project-collapse.

18. "Platform" and Hardware: Where will this be used and deployed?

19. Dependencies, Black-Box Services, & 'primative' tools
- It is probably broadly misunderstood that models are not the same as service-bundles:
-- input format and parts
-- output format
-- the loss of parameter setting




### Overall: Python vs. Rust Solution Comparison

(Note: This is very specific to this project; this is not commentary on python-data science in general.)

The whole python-LLM approach, for this specific case, is a classic production deployment nightmare. Everywhere you look there is a trail of gauntlets and quagmires extending endlessly into the future: scope, dependencies, cost, risks, maintainability, evaluation, reliability, general project-management, this is a disaster.

In comparison, the Rust-NLP solution is a glowing example of a practical application, flexible enough to survive all the real-world context questions that make the Python approach a cynical joke. 


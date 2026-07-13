#### Data Science Schema Map Study Project
# Schema Field Mapper

## Problem Statement

We have two database schemas from separate systems:
##### Dataset A is a MySQL schema for a legacy HR management system.
##### Dataset B is a MongoDB schema for a modern people platform.

The task is to build a pipeline that automatically maps fields in the source to semantically equivalent fields in the destination, accounting for all the fields being or unmapped.

The pipeline must output a single JSON file or object. For each field mapping, the JSON must include (other fields can be added optionally):
```text
source_field: source_field
destination_field: destination_field — use dot notation for nested paths, e.g. fullName.firstName
type_transform: type_transform — e.g. TINYINT(1) → Boolean or CHAR code → String enum
confidence: confidence
reasoning: reasoning — one plain-English sentence explaining the match
notes: notes — any value-transform logic required, or null
```

#### What approach? Pros and Cons
Depending on how and where this tool is used, it might be better to use an LLM, or might be better to use traditional NLP (Natural Language Processing).
Discuss at least two approaches.
Part 1: Make a python LLM-based solution.
Part 2: Make a GOFAI-NLP solution (ideally in a production safe language like Rust or Zig)
Try both approaches and write up your thoughts on strengths and weaknesses of each for possible future uses. Consider real world inputs, outputs, tech debt, costs, & maintainability.

As integrating the two in a sensible way may best fit many situations, what how would you design an integration of the strengths of both approaches? For example, depending on a given situation, would you tailor a both parts to the input provided here, or would the role of the LLM be to not work directly on the results by to design a new deterministic (auditable and consistant) system for any given schema inputs? 

Also, how might the output be more directly useful to the data-migration. Rather than producing more reports for people to read, could the system directly or indirectly support the migration itself (using python for migration to MongoDB is usually clean). 


## Optional Constraints:
1. You cannot pass both schemas to an LLM in a single prompt and receive a finished mapping.
2. The confidence, reasoning, notes report fields cannot be completed by generative AI, but must be meaningfully filled deterministically so as to be repeatable and auditable.


(Note: This is an arbitrary and artificial constraint, but consider how it might resemble real-world factors to work around and comment in your writeup.)


# Dataset A — Source Schema (MySQL)
Database: legacy_hrm   |   Type: MySQL (Relational)   |   Tables: emp_master, dept_info, locations
```
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

```

# Dataset B — Target Schema (MongoDB)
Database: people_platform   |   Type: MongoDB (Document)   |   Collections: employees, departments, locations
```
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

```

# Expected Output Format
Your pipeline must produce a JSON file or object that is compatible with this schema. The example below is partial. Your final output must cover all fields across all source tables.
```
{
  "version": 1,
  "source": "legacy_hrm (MySQL)",
  "destination": "people_platform (MongoDB)",
  "generated_at": "<ISO 8601 timestamp>",
  "tables": [
    {
      "source_table": "emp_master",
      "destination_collection": "employees",
      "confidence": 0.97,
      "reasoning": "Both represent the employee entity ...",
      "field_mappings": [
        {
          "source_field":       "emp_id",
          "destination_field":  "_id",
          "type_transform":     "INT -> ObjectId",
          "confidence":         0.91,
          "reasoning":          "Primary key maps to MongoDB _id; ID generation strategy is required.",
          "notes":              "Store original emp_id as legacy field for traceability."
        },
        {
          "source_field":       "f_name",
          "destination_field":  "fullName.firstName",
          "type_transform":     "VARCHAR -> String (nested path)",
          "confidence":         0.98,
          "reasoning":          "Flat field promoted into the fullName sub-document.",
          "notes":              null
        },
        {
          "source_field":       "rec_stat",
          "destination_field":  "employment.status",
          "type_transform":     "CHAR(1) code -> String enum",
          "confidence":         0.95,
          "reasoning":          "Single-char codes require a lookup transform to readable strings.",
          "notes":              "Transform: A -> active, I -> inactive, T -> terminated"
        },
        {
          "source_field":       "is_remote",
          "destination_field":  "employment.isRemote",
          "type_transform":     "TINYINT(1) -> Boolean",
          "confidence":         0.99,
          "reasoning":          "MySQL boolean integer pattern maps to MongoDB native Boolean.",
          "notes":              null
        }
        // ... all remaining fields
      ],
      "unmapped_source_fields": [],
      "unmapped_destination_fields": []
    }
    // ... dept_info -> departments
    // ... locations -> locations
  ]
}

```

# Deliverables

1. Working code
2. Generated output JSON for the two schema inputs above
3. Write-up about prompts and design decisions

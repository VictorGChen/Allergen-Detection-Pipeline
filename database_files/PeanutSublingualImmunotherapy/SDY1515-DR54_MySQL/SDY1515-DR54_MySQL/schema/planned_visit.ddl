DROP TABLE IF EXISTS planned_visit;

CREATE TABLE planned_visit
(
  
  planned_visit_accession VARCHAR(15) NOT NULL
    COMMENT "Primary key",
  
  end_rule VARCHAR(256)
    COMMENT "Conditions that define the end of a Planned Visit",
  
  max_start_day FLOAT
    COMMENT "Final day in the study timeline where the planned visit may occur",
  
  min_start_day FLOAT
    COMMENT "Initial day in the study timeline",
  
  name VARCHAR(256)
    COMMENT "Name or identifier",
  
  order_number INT NOT NULL
    COMMENT "Order of event",
  
  start_rule VARCHAR(256)
    COMMENT "Description of the conditions that define the beginning of a Planned Visit.",
  
  study_accession VARCHAR(15)
    COMMENT "Foreign key reference to the STUDY table",
  
  workspace_id INT NOT NULL
    COMMENT "Foreign key reference to WORKSPACE table",
  
  PRIMARY KEY (planned_visit_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Describes an STUDY indicated encounter with a SUBJECT.";

CREATE INDEX idx_planned_visit_workspace on planned_visit(workspace_id);


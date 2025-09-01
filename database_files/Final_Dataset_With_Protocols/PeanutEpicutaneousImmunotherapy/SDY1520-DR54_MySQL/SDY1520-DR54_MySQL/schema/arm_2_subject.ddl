DROP TABLE IF EXISTS arm_2_subject;

CREATE TABLE arm_2_subject
(
  
  arm_accession VARCHAR(15) NOT NULL
    COMMENT "Foreign key reference to ARMO_OR_COHORT table",
  
  subject_accession VARCHAR(15) NOT NULL
    COMMENT "Foreign key reference to SUBJECT table",
  
  age_event VARCHAR(40) NOT NULL DEFAULT 'Not Specified'
    COMMENT "Standardized value in the LK_AGE_EVENT table",
  
  age_event_specify VARCHAR(50)
    COMMENT "Reported value when the age event is not available in the LK_AGE_EVENT table",
  
  age_unit VARCHAR(25) NOT NULL DEFAULT 'Not Specified'
    COMMENT "Standardized value in the LK_TIME_UNIT table",
  
  max_subject_age FLOAT
    COMMENT "Maximum numeric age of the subject",
  
  min_subject_age FLOAT
    COMMENT "Minumum numeric age of the subject",
  
  subject_phenotype VARCHAR(200)
    COMMENT "Reported value",

  subject_location VARCHAR(50)
    COMMENT "Subject geographic location.",

  max_subject_age_in_years FLOAT
    COMMENT "Maximum numeric age of the subject in years",
  
  min_subject_age_in_years FLOAT
    COMMENT "Minumum numeric age of the subject in years",
  
  PRIMARY KEY (arm_accession, subject_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Join table that associates Subject with ARM_OR_COHORT table records.";

CREATE INDEX idx_arm_2_subject_1 on arm_2_subject(subject_accession,arm_accession);


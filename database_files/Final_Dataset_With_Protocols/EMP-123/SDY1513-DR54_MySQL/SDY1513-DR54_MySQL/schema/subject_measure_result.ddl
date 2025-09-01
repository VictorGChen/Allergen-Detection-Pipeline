DROP TABLE IF EXISTS subject_measure_result;

CREATE TABLE subject_measure_result
(
  
  subject_measure_res_accession VARCHAR(15) NOT NULL
    COMMENT "Primary key",
  
  centraltendencymeasurevalue VARCHAR(40)
    COMMENT "Value of the central tendency measure defined in the SUBJECT_MEASURE_DEFINITION record",
  
  datavalue VARCHAR(40)
    COMMENT "Value calculated or recorded for this measure.",
  
  dispersionmeasurevalue VARCHAR(40)
    COMMENT "Value of the dispersion measure defined in the SUBJECT_MEASURE_DEFINITION record",
  
  study_accession VARCHAR(15)
    COMMENT "Foreign key reference to the STUDY table",
  
  study_day FLOAT
    COMMENT "Reported value",
  
  subject_accession VARCHAR(15) NOT NULL
    COMMENT "Foreign key reference to the SUBJECT table",
  
  subject_measure_accession VARCHAR(15) NOT NULL
    COMMENT "Foreign key reference SUBJECT_MEASURE_DEFINITION table",
  
  time_of_day VARCHAR(40)
    COMMENT "Reported value",
  
  workspace_id INT NOT NULL
    COMMENT "Foreign key reference to WORKSPACE table",
  
  year_of_measure DATE
    COMMENT "Year that the measure was taken or recorded",
  
  PRIMARY KEY (subject_measure_res_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Contains the computed results of SUBJECT_MEASURE.";

CREATE INDEX idx_subject_measure_result_study on subject_measure_result(study_accession);
CREATE INDEX idx_subject_measure_result_subject on subject_measure_result(subject_accession);
CREATE INDEX idx_subject_measure_result_workspace on subject_measure_result(workspace_id);


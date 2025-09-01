DROP TABLE IF EXISTS subject_measure_definition;

CREATE TABLE subject_measure_definition
(
  
  subject_measure_accession VARCHAR(15) NOT NULL
    COMMENT "Primary key",
  
  algorithm VARCHAR(1024)
    COMMENT "Formula for calculating secondary measures",
  
  description VARCHAR(4000)
    COMMENT "Free text to expand upon details",
  
  measureofcentraltendency VARCHAR(40)
    COMMENT "Whether mean, median or mode was utilized when there is a central tendency calculation performed",
  
  measureofdispersion VARCHAR(40)
    COMMENT "Variability or spread in a variable or probability calculation, such as variance, standard deviation, or interquartile range",
  
  measuretype VARCHAR(40)
    COMMENT "Description of the nature of the measure",
  
  name VARCHAR(125)
    COMMENT "Name or identifier",
  
  outcometype VARCHAR(70)
    COMMENT "A characterization of the type of measurement being described",
  
  study_accession VARCHAR(15)
    COMMENT "Foreign key reference to the STUDY table",
  
  timeframe VARCHAR(256)
    COMMENT "A textual description of the timeframe involved in this measure (e.g. all days of the year)",
  
  unitofmeasure VARCHAR(40)
    COMMENT "Reported value",
  
  workspace_id INT NOT NULL
    COMMENT "Foreign key reference to WORKSPACE table",
  
  PRIMARY KEY (subject_measure_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Contains computed measures for a SUBJECT.";


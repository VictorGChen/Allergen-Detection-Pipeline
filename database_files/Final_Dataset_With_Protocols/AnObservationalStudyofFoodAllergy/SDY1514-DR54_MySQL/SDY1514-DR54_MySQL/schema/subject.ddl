DROP TABLE IF EXISTS subject;

CREATE TABLE subject
(
  
  subject_accession VARCHAR(15) NOT NULL
    COMMENT "Primary key",
  
  ancestral_population VARCHAR(100)
    COMMENT "Foreign key reference to LK_ANCESTRAL_POPULATION table",
  
  description VARCHAR(4000)
    COMMENT "Free text to expand upon details",
  
  ethnicity VARCHAR(50) DEFAULT 'Not Specified'
    COMMENT "Standardized value in LK_ETHNICITY table",
  
  gender VARCHAR(20) NOT NULL DEFAULT 'Not Specified'
    COMMENT "Standardized value in LK_GENDER table",
  
  race VARCHAR(50) DEFAULT 'Not Specified'
    COMMENT "Standardized value in LK_RACE table",
  
  race_specify VARCHAR(1000)
    COMMENT "Entered by the data provider when term is not in the controlled vocabulary",
  
  species VARCHAR(50) NOT NULL
    COMMENT "Standardized value in LK_SPECIES table",
  
  strain VARCHAR(50)
    COMMENT "Reported value",
  
  strain_characteristics VARCHAR(500)
    COMMENT "Reported value",
  
  workspace_id INT NOT NULL
    COMMENT "Foreign key reference to WORKSPACE table",
  
  PRIMARY KEY (subject_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Humans or animals that are the focus of a STUDY.";

CREATE INDEX idx_subject_gender on subject(gender);
CREATE INDEX idx_subject_race on subject(race);
CREATE INDEX idx_subject_species on subject(species);
CREATE INDEX idx_subject_workspace on subject(workspace_id);

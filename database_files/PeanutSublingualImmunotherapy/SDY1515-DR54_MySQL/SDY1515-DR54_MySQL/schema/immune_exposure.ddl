DROP TABLE IF EXISTS immune_exposure;

CREATE TABLE immune_exposure
(
  
  exposure_accession VARCHAR(15) NOT NULL
    COMMENT "The primary key for the Immune Exposure table",
  
  arm_accession VARCHAR(15) NOT NULL
    COMMENT "The study arm to which the subject is associated",

  disease_ontology_id VARCHAR(100)
    COMMENT "The disease ontology ID associated with the disease",

  disease_preferred VARCHAR(250)
    COMMENT "The preferred disease associated with the exposure",

  disease_reported VARCHAR(550)
    COMMENT "This indicated the specific disease of the host associated with the exposure",

  disease_stage_preferred VARCHAR(50)
    COMMENT "The preferred disease stage associate with the the exposure. The Disease Stage reflects the stage of the disease at the time the experimental data was generated.",

  disease_stage_reported VARCHAR(100)
    COMMENT "This provides a broad classification of how the disease has progressed. The Disease Stage reflects the stage of the disease at the time the experimental data was generated.",

  exposure_material_id VARCHAR(100)
    COMMENT "The NCBI or Vaccine Ontology ID associated with the exposure material.",

  exposure_material_preferred VARCHAR(200)
    COMMENT "The preferred exposure material associated with the process",

  exposure_material_reported VARCHAR(200)
    COMMENT "This describes what substance(s) the host is exposed to and/or develops immune reactions to as part of the exposure process.",

  exposure_process_preferred VARCHAR(100)
    COMMENT "The preferred exposure process",

  exposure_process_reported VARCHAR(100) NOT NULL
    COMMENT "This identifies the type of process through which a host is exposed and the type of evidence for that exposure to have happened, which are tightly intertwined. This is the only element of the four that is always mandatory.",

  subject_accession VARCHAR(15)
    COMMENT "The subject associated with the Immune exposure record",
  
  PRIMARY KEY (exposure_accession)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Captures results of immune exposure.";

CREATE INDEX idx_immune_arm_accession on immune_exposure(arm_accession);
CREATE INDEX idx_immune_subject_accession on immune_exposure(subject_accession);

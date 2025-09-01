DROP TABLE IF EXISTS lk_subject_location;

CREATE TABLE lk_subject_location
(
  
  name VARCHAR(50) NOT NULL
    COMMENT "Name or identifier",

  description VARCHAR(4000) NOT NULL
    COMMENT "Free text to expand upon details",
  
  link VARCHAR(2000)
    COMMENT "Link to URL",
  
  PRIMARY KEY (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8
  COMMENT "Controlled vocabulary for subject location";

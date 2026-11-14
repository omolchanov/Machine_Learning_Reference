-- Local dev user for MySQL Workbench / dbt (Docker only).

CREATE USER IF NOT EXISTS 'ml_ref_dev'@'%'
  IDENTIFIED WITH caching_sha2_password BY 'ml_ref_dev_pass';

GRANT ALL PRIVILEGES ON dbt_sample.* TO 'ml_ref_dev'@'%';
GRANT SHOW DATABASES ON *.* TO 'ml_ref_dev'@'%';
FLUSH PRIVILEGES;
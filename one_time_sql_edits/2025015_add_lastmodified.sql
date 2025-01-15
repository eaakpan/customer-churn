ALTER TABLE churnset.customers ADD COLUMN last_modified TIMESTAMP;
ALTER TABLE churnset.customers ALTER COLUMN last_modified SET DEFAULT now();

ALTER TABLE churnset.customers ADD COLUMN created_at TIMESTAMP;
ALTER TABLE churnset.customers ALTER COLUMN created_at SET DEFAULT now();

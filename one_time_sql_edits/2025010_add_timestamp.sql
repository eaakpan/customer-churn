ALTER TABLE churnset.customers ADD COLUMN created_att TIMESTAMP;
ALTER TABLE churnset.customers ALTER COLUMN created_at SET DEFAULT now();

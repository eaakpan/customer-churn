
ALTER TABLE churnset.customers DROP CONSTRAINT customers_seniorcitizen_check;

ALTER TABLE churnset.customers ALTER COLUMN SeniorCitizen TYPE VARCHAR(3) USING SeniorCitizen::VARCHAR(3)

UPDATE churnset.customers
SET SeniorCitizen = 'Yes'
WHERE SeniorCitizen = '1';

UPDATE churnset.customers
SET SeniorCitizen = 'No'
WHERE SeniorCitizen = '0';

ALTER TABLE churnset.customers
ADD CONSTRAINT customers_seniorcitizen_check CHECK (SeniorCitizen IN ('Yes', 'No'));




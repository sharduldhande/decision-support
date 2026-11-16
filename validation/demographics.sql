-- Age and sex for every admission in the validation sample. Run in BigQuery (project mimic-iv-441820).
-- Export the result to validation/data/demographics.csv (git-ignored). No note text is selected.
--
-- MIMIC-IV gives each patient an anchor_age in a shifted anchor_year, so age at this admission is
-- anchor_age plus the years between anchor_year and the (equally shifted) admission year.
-- Patients over 89 are recorded as anchor_age 91, so ages of 90 and above are capped, not exact.

SELECT
  s.note_id,
  s.subject_id,
  s.hadm_id,
  s.label,
  p.gender AS sex,                                                        -- 'M' or 'F'
  p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
  p.anchor_age >= 91 AS age_capped                                        -- true: real age is over 89
FROM `mimic-iv-441820.mimic_project.ds_validation_sample` s
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
ORDER BY s.label, s.note_id;

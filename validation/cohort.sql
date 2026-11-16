-- decision-support validation cohort. Run in BigQuery (project mimic-iv-441820) on 2026-11-15.
-- NSTEMI / unstable angina admissions whose discharge note reports cath findings,
-- labelled by what was done in the same admission: CABG, PCI, or neither (MEDICAL).
-- Creates mimic-iv-441820.mimic_project.ds_validation_cohort (all eligible admissions)
-- and ds_validation_sample (500 CABG + 500 PCI, one admission per patient).
-- Scans the discharge note text once: a few GB, inside the free 1 TB/month.

CREATE OR REPLACE TABLE `mimic-iv-441820.mimic_project.ds_validation_cohort` AS
WITH dx AS (
  SELECT
    hadm_id,
    -- NSTEMI: ICD-10 I21.4, ICD-9 410.7x. Unstable angina: ICD-10 I20.0, ICD-9 411.1.
    LOGICAL_OR(
      (icd_version = 10 AND (icd_code LIKE 'I214%' OR icd_code LIKE 'I200%'))
      OR (icd_version = 9 AND (icd_code LIKE '4107%' OR icd_code LIKE '4111%'))
    ) AS nstemi_ua,
    -- STEMI (ICD-10 I21.0-I21.3, ICD-9 410.0-410.6 and 410.8): the app declines these.
    LOGICAL_OR(
      (icd_version = 10 AND REGEXP_CONTAINS(icd_code, r'^I21[0-3]'))
      OR (icd_version = 9 AND REGEXP_CONTAINS(icd_code, r'^410[0-68]'))
    ) AS stemi
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
  GROUP BY hadm_id
),
px AS (
  SELECT
    hadm_id,
    -- CABG: ICD-10-PCS 021 (bypass heart), ICD-9 36.1x.
    LOGICAL_OR((icd_version = 10 AND icd_code LIKE '021%')
            OR (icd_version = 9 AND icd_code LIKE '361%')) AS cabg,
    -- PCI: ICD-10-PCS 027 (dilation of coronary arteries), ICD-9 00.66, 36.06, 36.07, 36.09.
    LOGICAL_OR((icd_version = 10 AND icd_code LIKE '027%')
            OR (icd_version = 9 AND icd_code IN ('0066', '3606', '3607', '3609'))) AS pci,
    -- Diagnostic cath / coronary angiography: ICD-10-PCS B21 (fluoroscopy of coronary
    -- arteries) or 4A023N (cardiac cath sampling and pressure); ICD-9 37.21-37.23, 88.55-88.57.
    LOGICAL_OR((icd_version = 10 AND (icd_code LIKE 'B21%' OR icd_code LIKE '4A023N%'))
            OR (icd_version = 9 AND icd_code IN ('3721', '3722', '3723', '8855', '8856', '8857'))) AS cath
  FROM `physionet-data.mimiciv_3_1_hosp.procedures_icd`
  GROUP BY hadm_id
),
notes AS (
  -- Last discharge note per admission, kept only if it states per-vessel stenosis
  -- (same pattern as the local 5k-sample count: about 2.8% of all notes).
  SELECT * EXCEPT (rn)
  FROM (
    SELECT n.note_id, n.subject_id, n.hadm_id, n.text,
           ROW_NUMBER() OVER (PARTITION BY n.hadm_id ORDER BY n.note_seq DESC) AS rn
    FROM `physionet-data.mimiciv_note.discharge` n
  )
  WHERE rn = 1
    AND REGEXP_CONTAINS(text, r'(?i)(cardiac cath|catheterization|coronary angiogra)')
    AND REGEXP_CONTAINS(text, r'(?i)(LAD|LMCA|left main|RCA|LCx|circumflex)[^\n]{0,60}?\d{2,3}\s?%')
)
SELECT
  n.note_id, n.subject_id, n.hadm_id,
  a.admittime, a.dischtime,
  CASE
    WHEN px.cabg AND px.pci THEN 'BOTH'
    WHEN px.cabg THEN 'CABG'
    WHEN px.pci THEN 'PCI'
    ELSE 'MEDICAL'
  END AS label,
  IFNULL(px.cath, FALSE) AS cath_coded,
  n.text
FROM notes n
JOIN dx USING (hadm_id)
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
LEFT JOIN px USING (hadm_id)
WHERE dx.nstemi_ua AND NOT dx.stemi;


-- 1,000-case sample: 500 CABG + 500 PCI, one admission per patient, deterministic.
-- BOTH is dropped (staged procedures muddy the label); MEDICAL stays in the cohort
-- table for later use.
CREATE OR REPLACE TABLE `mimic-iv-441820.mimic_project.ds_validation_sample` AS
WITH one_per_patient AS (
  SELECT * EXCEPT (rn)
  FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY subject_id, label
                              ORDER BY FARM_FINGERPRINT(CAST(hadm_id AS STRING))) AS rn
    FROM `mimic-iv-441820.mimic_project.ds_validation_cohort`
    WHERE label IN ('CABG', 'PCI')
  )
  WHERE rn = 1
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY label
                            ORDER BY FARM_FINGERPRINT(CAST(hadm_id AS STRING))) AS pick
  FROM one_per_patient
)
SELECT * EXCEPT (pick)
FROM ranked
WHERE pick <= 500
  -- a patient can appear once as CABG and once as PCI; drop them
  AND subject_id NOT IN (
    SELECT subject_id FROM ranked WHERE pick <= 500
    GROUP BY subject_id HAVING COUNT(*) > 1
  );


-- Counts only (no note text).
SELECT label, cath_coded, COUNT(*) AS admissions, COUNT(DISTINCT subject_id) AS patients
FROM `mimic-iv-441820.mimic_project.ds_validation_cohort`
GROUP BY label, cath_coded
ORDER BY label, cath_coded;

SELECT label, COUNT(*) AS n
FROM `mimic-iv-441820.mimic_project.ds_validation_sample`
GROUP BY label;

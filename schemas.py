from typing import ClassVar, Literal

from pydantic import BaseModel, Field, computed_field

Severity = Literal["significant", "intermediate", "non_significant", "unknown"]


class Segment(BaseModel):
    SIGNIFICANT_PCT: ClassVar[int] = 70
    INTERMEDIATE_PCT: ClassVar[int] = 40

    stenosis_pct: list[int] | None = Field(
        default=None,
        description=(
            "Stated stenosis percentage(s): [30] for a single value, [30, 40] for a range like "
            "'30-40%'. Preserve ranges exactly; never round or pick one value. "
            "null if no percentage is stated."
        ),
    )
    stenosis_qualifier: Literal[
        "normal", "mild", "mild_plaque", "moderate", "severe", "significant", "total_occlusion", "not_described"
    ] = Field(
        default="not_described",
        description=(
            "Qualitative description. 'normal' when the report explicitly says normal or no disease "
            "(also for every segment of a vessel described as normal overall). "
            "'significant' only when the report literally uses the word 'significant' for this segment, "
            "not for a plain 'stenosis'. "
            "'not_described' only when a percentage is given without a qualitative word, "
            "or the segment is not mentioned."
        ),
    )

    @computed_field
    @property
    def severity(self) -> Severity:
        if self.stenosis_qualifier in ("severe", "total_occlusion"):
            return "significant"
        if self.stenosis_pct:
            pct = max(self.stenosis_pct)
            if pct >= self.SIGNIFICANT_PCT:
                return "significant"
            if pct >= self.INTERMEDIATE_PCT:
                return "intermediate"
            return "non_significant"
        if self.stenosis_qualifier == "significant":
            return "significant"
        if self.stenosis_qualifier == "moderate":
            return "intermediate"
        if self.stenosis_qualifier in ("normal", "mild", "mild_plaque"):
            return "non_significant"
        return "unknown"


class LeftMainSegment(Segment):
    SIGNIFICANT_PCT: ClassVar[int] = 50


class LeftMain(BaseModel):
    ostial: LeftMainSegment = Field(default_factory=LeftMainSegment)
    mid: LeftMainSegment = Field(default_factory=LeftMainSegment)
    distal: LeftMainSegment = Field(default_factory=LeftMainSegment)


class Vessel(BaseModel):

    proximal: Segment = Field(default_factory=Segment)
    mid: Segment = Field(default_factory=Segment)
    distal: Segment = Field(default_factory=Segment)


class DominantVessel(Vessel):
    dominant: bool = Field(
        default=False,
        description=(
            "From the stated dominance. Right dominant: RCA true, LCx false. "
            "Left dominant: LCx true, RCA false. Codominant: both true."
        ),
    )


class Branches(BaseModel):
    diagonal_1: Segment = Field(default_factory=Segment)
    diagonal_2: Segment = Field(default_factory=Segment)
    om_1: Segment = Field(default_factory=Segment)
    om_2: Segment = Field(default_factory=Segment)
    ramus: Segment = Field(default_factory=Segment)
    pda: Segment = Field(default_factory=Segment)
    plb: Segment = Field(default_factory=Segment)


class CoronaryAnatomy(BaseModel):
    dominance: Literal["right", "left", "codominant", "unknown"] = "unknown"
    left_main: LeftMain = Field(default_factory=LeftMain)
    lad: Vessel = Field(default_factory=Vessel)
    lcx: DominantVessel = Field(default_factory=DominantVessel)
    rca: DominantVessel = Field(default_factory=DominantVessel)
    branches: Branches = Field(default_factory=Branches)


class ClinicalContext(BaseModel):
    comorbidities: list[str] = Field(
        default_factory=list,
        description="Non-cardiac conditions as listed, e.g. ['HTN', 'DM', 'CKD']. Empty if none listed.",
    )
    cardiac_diagnosis: str | None = Field(
        default=None, description="Primary cardiac indication, e.g. 'CAD - ACS - STEMI'. null if not stated."
    )
    presentation: Literal[
        "NSTEMI", "unstable_angina", "ACS_unspecified", "STEMI", "posterior_MI",
        "stable", "non_coronary", "not_stated",
    ] = Field(
        default="not_stated",
        description=(
            "Clinical presentation from the report's indication or diagnosis. "
            "'posterior_MI' for posterior wall MI (PWMI), even if labelled STEMI. "
            "'ACS_unspecified' when the report says ACS without saying STEMI, NSTEMI or unstable angina. "
            "'stable' for elective work-up: chest pain evaluation, positive stress test, known stable CAD. "
            "'non_coronary' if the report has no coronary angiography (e.g. valve, congenital, right heart only). "
            "'not_stated' if the report gives no indication."
        ),
    )
    valvular_disease: str | None = Field(
        default=None, description="Valvular findings, e.g. 'SEVERE MR'. null if not stated."
    )
    rhythm: str | None = Field(default=None, description="Cardiac rhythm, e.g. 'SR', 'AF'. null if not stated.")
    lv_function: str | None = Field(
        default=None, description="LV function, e.g. 'MILD LV DYSFUNCTION', 'LVEF 35%'. null if not stated."
    )


class Lesion(BaseModel):
    vessel: str
    segment: str
    severity: Severity
    stenosis_pct: list[int] | None
    stenosis_qualifier: str


VESSEL_NAMES = {"left_main": "Left Main", "lad": "LAD", "lcx": "LCx", "rca": "RCA"}


class AngiographicDiagnosis(BaseModel):
    summary: str | None = Field(
        default=None, description="The report's final diagnosis line, e.g. 'CAD - DVD'. null if not stated."
    )
    vessel_disease_count: int | None = Field(
        default=None,
        description="Number of diseased vessels from the diagnosis (SVD=1, DVD=2, TVD=3). null if not stated.",
    )


class Findings(BaseModel):

    clinical_context: ClinicalContext = Field(default_factory=ClinicalContext)
    coronary_anatomy: CoronaryAnatomy = Field(default_factory=CoronaryAnatomy)
    angiographic_diagnosis: AngiographicDiagnosis = Field(default_factory=AngiographicDiagnosis)

    @computed_field
    @property
    def significant_vessels(self) -> list[str]:
        anatomy = self.coronary_anatomy
        dominant = "LCx" if anatomy.dominance == "left" else "RCA"
        parent = {"diagonal_1": "LAD", "diagonal_2": "LAD", "om_1": "LCx", "om_2": "LCx",
                  "ramus": "Ramus", "pda": dominant, "plb": dominant}
        vessels = [
            VESSEL_NAMES[name] for name in VESSEL_NAMES
            if any(isinstance(seg, Segment) and seg.severity == "significant"
                   for _, seg in getattr(anatomy, name))
        ]
        vessels += [parent[branch] for branch, seg in anatomy.branches if seg.severity == "significant"]
        return list(dict.fromkeys(vessels))

    @computed_field
    @property
    def lesions(self) -> list[Lesion]:
        anatomy = self.coronary_anatomy
        segments = [
            (VESSEL_NAMES[name], seg_name, seg)
            for name in VESSEL_NAMES
            for seg_name, seg in getattr(anatomy, name)
            if isinstance(seg, Segment)
        ]
        segments += [
            (branch.replace("_", " ").title(), "", seg) for branch, seg in anatomy.branches
        ]
        return [
            Lesion(vessel=vessel, segment=seg_name, severity=seg.severity,
                   stenosis_pct=seg.stenosis_pct, stenosis_qualifier=seg.stenosis_qualifier)
            for vessel, seg_name, seg in segments
            if seg.severity in ("significant", "intermediate")
        ]


class Citation(BaseModel):
    section_title: str = Field(description="Exact section title as it appears in the provided guideline text")
    class_of_recommendation: Literal["1", "2a", "2b", "3: No Benefit", "3: Harm"] | None = Field(
        default=None,
        description="Copied from a provided 'Recommendation (COR ..., LOE ...)'; null if citing prose",
    )
    level_of_evidence: Literal["A", "B-R", "B-NR", "C-LD", "C-EO"] | None = Field(
        default=None,
        description="Copied from a provided 'Recommendation (COR ..., LOE ...)'; null if citing prose",
    )
    guideline_text: str = Field(description="Brief neutral summary of what the guideline states")
    patient_relevance: str = Field(description="Why this clause applies to this specific patient")


class Decision(BaseModel):

    recommendation: Literal[
        "PCI", "CABG", "PCI preferred", "CABG preferred",
        "Either acceptable", "Medical management", "Insufficient data",
    ]
    confidence: Literal["strong", "moderate", "weak"]
    primary_rationale: str = Field(description="2-3 sentence plain-language explanation")
    supporting_citations: list[Citation] = Field(
        description="At least one citation, preferring the provided formal recommendations"
    )
    conditions_favouring_pci: list[str] = Field(description="Patient factors favouring PCI; empty only if none")
    conditions_favouring_cabg: list[str] = Field(description="Patient factors favouring CABG; empty only if none")
    conditions_favouring_medical_management: list[str] = Field(
        description="Patient factors favouring medical management; empty only if none"
    )
    heart_team_considerations: str = Field(description="Additional note for the multidisciplinary Heart Team")
    urgent_issues: str | None = Field(default=None, description="Anything needing urgent attention; null if none")

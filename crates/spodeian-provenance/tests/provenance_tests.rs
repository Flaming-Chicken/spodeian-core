use spodeian_provenance::{
    LedgerRecord, LicenseVerifier, ProvenanceError, SourceOrigin, render_attributions_txt,
    validate_ledger,
};

fn sample_record(id: &str, license: &str, origin: SourceOrigin, warranty: bool) -> LedgerRecord {
    LedgerRecord {
        id: id.to_string(),
        title: format!("Rain Sample {id}"),
        author: "Contributor".to_string(),
        origin,
        license_spdx: license.to_string(),
        source_url: Some(format!("https://example.org/{id}")),
        sha256: format!("hash-{id}"),
        duration_secs: 12.5,
        sample_rate_hz: 48_000,
        surface: Some("tin_roof".to_string()),
        rights_warranty: warranty,
        ingested_at_unix: 1_700_000_000,
    }
}

#[test]
fn test_whole_token_license_verifier_avoids_substring_false_positives() {
    let v = LicenseVerifier::commercial_safe_default();
    assert!(v.is_allowed("CC0-1.0"));
    assert!(v.is_allowed("cc_by_4.0"));
    assert!(v.is_allowed("CC BY 3.0"));
    assert!(v.is_allowed("Synthetic"));
    assert!(v.is_allowed("Spodeian-D4"));

    // Forbidden tokens must be rejected cleanly
    assert!(!v.is_allowed("CC-BY-NC-4.0"));
    assert!(!v.is_allowed("CC-BY-NC-SA-4.0"));
    assert!(!v.is_allowed("CC-BY-ND-4.0"));
    assert!(!v.is_allowed("All Rights Reserved"));
    assert!(!v.is_allowed(""));
}

#[test]
fn test_validate_ledger_enforces_warranty_duplicates_and_licenses() {
    let v = LicenseVerifier::default();
    let ok = vec![
        sample_record("a1", "CC0-1.0", SourceOrigin::Freesound, false),
        sample_record("a2", "Spodeian-D4", SourceOrigin::ContributorGrant, true),
    ];
    validate_ledger(&ok, &v).expect("valid ledger");

    let missing_warranty = vec![sample_record(
        "a3",
        "Spodeian-D4",
        SourceOrigin::ContributorGrant,
        false,
    )];
    assert!(matches!(
        validate_ledger(&missing_warranty, &v),
        Err(ProvenanceError::MissingRightsWarranty(_))
    ));

    let bad_license = vec![sample_record(
        "a4",
        "CC-BY-NC-4.0",
        SourceOrigin::Freesound,
        false,
    )];
    assert!(matches!(
        validate_ledger(&bad_license, &v),
        Err(ProvenanceError::DisallowedLicense { .. })
    ));
}

#[test]
fn test_render_attributions_filters_and_sorts_deterministically() {
    let v = LicenseVerifier::default();
    let records = vec![
        sample_record("z9", "CC-BY-4.0", SourceOrigin::Freesound, false),
        sample_record("a1", "CC0-1.0", SourceOrigin::Freesound, false),
        sample_record("b2", "CC-BY-3.0", SourceOrigin::Freesound, false),
    ];

    let cc_by_only = render_attributions_txt(&records, &v, false);
    assert!(!cc_by_only.contains("[a1]"));
    let b2_pos = cc_by_only.find("[b2]").expect("b2 present");
    let z9_pos = cc_by_only.find("[z9]").expect("z9 present");
    assert!(b2_pos < z9_pos);
}

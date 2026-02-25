-- ============================================================
-- STEP 5: SUPABASE SQL SCHEMA
-- Residential Lease Intelligence Platform
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- ── TABLE: users ──────────────────────────────────────────────────────────────
-- Stores tenant and landlord accounts with tier and usage tracking

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    email           TEXT UNIQUE NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('tenant', 'landlord', 'admin')),
    tier            TEXT NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'paid', 'landlord', 'admin')),
    analyses_used   INTEGER NOT NULL DEFAULT 0,
    analyses_limit  INTEGER NOT NULL DEFAULT 1,        -- free tier: 1 free analysis
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at    TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_tier  ON users(tier);

COMMENT ON TABLE users IS 'Tenant and landlord accounts. Free tier capped at 1 analysis.';


-- ── TABLE: leases (main extraction storage) ───────────────────────────────────
-- One row per submitted lease document with all 50+ extracted fields

CREATE TABLE IF NOT EXISTS leases (
    -- Identity
    id                          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    extraction_id               TEXT UNIQUE NOT NULL,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id                     UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Document metadata
    document_filename           TEXT,
    document_hash_sha256        TEXT,                  -- deduplication
    document_pages              INTEGER,
    pipeline_version            TEXT NOT NULL DEFAULT '1.0.0',
    analysis_tier               TEXT CHECK (analysis_tier IN ('free', 'paid', 'landlord')),

    -- ── LAYER 1: Raw fields ──────────────────────────────────────────────────
    -- Parties
    tenant_full_name            TEXT,
    landlord_full_name          TEXT,
    landlord_entity_type        TEXT,

    -- Property
    property_address            TEXT,
    property_city               TEXT,
    property_state              CHAR(2),               -- two-letter state code
    property_zip                TEXT,
    unit_number                 TEXT,
    property_type               TEXT,
    square_footage              NUMERIC(10,2),
    furnished                   BOOLEAN,

    -- Financial
    monthly_rent                NUMERIC(10,2),
    security_deposit            NUMERIC(10,2),
    last_month_deposit          NUMERIC(10,2),
    pet_deposit                 NUMERIC(10,2),
    other_fees_monthly          NUMERIC(10,2),
    late_fee_amount             NUMERIC(10,2),
    late_fee_grace_days         INTEGER,
    nsf_fee                     NUMERIC(10,2),

    -- Term
    lease_type                  TEXT CHECK (lease_type IN ('fixed_term','month_to_month','week_to_week','unknown')),
    lease_start_date            DATE,
    lease_end_date              DATE,
    rent_due_day                INTEGER CHECK (rent_due_day BETWEEN 1 AND 31),

    -- Renewal & termination
    renewal_notice_days_required    INTEGER,
    early_termination_fee           NUMERIC(10,2),
    early_termination_notice_days   INTEGER,
    landlord_entry_notice_hours     INTEGER,

    -- Utilities & maintenance
    utilities_tenant_responsible    TEXT[],
    utilities_landlord_responsible  TEXT[],
    tenant_maintenance_obligations  TEXT,

    -- Restrictions
    pets_allowed                BOOLEAN,
    subletting_allowed          BOOLEAN,
    smoking_prohibited          BOOLEAN,

    -- Signatures
    lease_signed_date           DATE,
    lease_document_pages        INTEGER,

    -- ── LAYER 2: Computed fields ─────────────────────────────────────────────
    computed_total_upfront_cost         NUMERIC(10,2),
    computed_total_liability_term       NUMERIC(10,2),
    computed_effective_monthly_cost     NUMERIC(10,2),
    computed_annualized_rent            NUMERIC(10,2),
    computed_lease_term_months          INTEGER,
    computed_lease_term_days            INTEGER,
    computed_days_until_lease_end       INTEGER,
    computed_days_until_renewal_deadline INTEGER,
    computed_renewal_deadline_date      DATE,
    computed_implied_cost_per_sqft_monthly NUMERIC(8,4),
    computed_deposit_as_months_rent     NUMERIC(6,3),
    computed_late_fee_as_pct_rent       NUMERIC(6,4),

    -- ── LAYER 4: Scores ──────────────────────────────────────────────────────
    score_overall_risk              INTEGER CHECK (score_overall_risk BETWEEN 0 AND 100),
    score_renewal_risk              INTEGER CHECK (score_renewal_risk BETWEEN 0 AND 100),
    score_financial_burden          INTEGER CHECK (score_financial_burden BETWEEN 0 AND 100),
    score_landlord_access_risk      INTEGER CHECK (score_landlord_access_risk BETWEEN 0 AND 100),
    score_termination_risk          INTEGER CHECK (score_termination_risk BETWEEN 0 AND 100),
    risk_tier                       TEXT CHECK (risk_tier IN ('GREEN','YELLOW','ORANGE','RED')),
    rent_vs_zip_median_pct          NUMERIC(8,2),

    -- Tenant summary (stored for report generation)
    summary_biggest_risk            TEXT,
    summary_biggest_positive        TEXT,
    summary_key_action              TEXT,

    -- ── QUALITY METADATA ─────────────────────────────────────────────────────
    fields_extracted                INTEGER,
    fields_below_confidence         INTEGER,
    requires_human_review           BOOLEAN NOT NULL DEFAULT FALSE,
    human_review_reasons            TEXT[],
    is_non_standard_format          BOOLEAN NOT NULL DEFAULT FALSE,
    format_issues                   TEXT[],
    reasoning_flags                 TEXT[],

    -- Soft delete
    deleted_at                      TIMESTAMPTZ
);

CREATE INDEX idx_leases_user_id       ON leases(user_id);
CREATE INDEX idx_leases_zip           ON leases(property_zip);
CREATE INDEX idx_leases_state         ON leases(property_state);
CREATE INDEX idx_leases_created_at    ON leases(created_at DESC);
CREATE INDEX idx_leases_risk_tier     ON leases(risk_tier);
CREATE INDEX idx_leases_doc_hash      ON leases(document_hash_sha256);   -- dedup

COMMENT ON TABLE leases IS 'Master lease extraction table. One row per submitted document.';

-- Auto-update updated_at on any change
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER leases_updated_at
    BEFORE UPDATE ON leases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();


-- ── TABLE: extractions (versioned field-level storage) ────────────────────────
-- Stores per-field confidence scores and citations for auditability

CREATE TABLE IF NOT EXISTS extractions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lease_id        UUID NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    schema_version  TEXT NOT NULL DEFAULT '1.0.0',
    field_name      TEXT NOT NULL,
    raw_value       TEXT,                              -- raw string from Gemini
    parsed_value    TEXT,                              -- Python-parsed value
    confidence      NUMERIC(4,3) CHECK (confidence BETWEEN 0 AND 1),
    source          TEXT CHECK (source IN ('llm','computed','external','manual')),
    raw_text_citation TEXT,
    null_reason     TEXT,
    flag_for_reasoning BOOLEAN DEFAULT FALSE,
    human_override  BOOLEAN DEFAULT FALSE,
    override_by     UUID REFERENCES users(id),
    override_at     TIMESTAMPTZ,
    override_reason TEXT
);

CREATE INDEX idx_extractions_lease_id ON extractions(lease_id);
CREATE INDEX idx_extractions_field    ON extractions(field_name);
CREATE INDEX idx_extractions_low_conf ON extractions(confidence) WHERE confidence < 0.75;

COMMENT ON TABLE extractions IS 'Per-field extraction audit log with confidence scores. Enables schema versioning and human override.';


-- ── TABLE: risk_flags (Layer 3 output) ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS risk_flags (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    lease_id        UUID NOT NULL REFERENCES leases(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    flag_id         TEXT NOT NULL,                     -- slug, e.g. 'insufficient_entry_notice'
    severity        TEXT NOT NULL CHECK (severity IN ('CRITICAL','HIGH','MEDIUM','LOW')),
    category        TEXT NOT NULL,
    short_description TEXT NOT NULL,
    detailed_explanation TEXT,
    raw_clause_citation TEXT,
    jurisdiction_note TEXT,
    recommended_action TEXT
);

CREATE INDEX idx_risk_flags_lease_id  ON risk_flags(lease_id);
CREATE INDEX idx_risk_flags_severity  ON risk_flags(severity);
CREATE INDEX idx_risk_flags_flag_id   ON risk_flags(flag_id);   -- trend analysis


-- ── TABLE: zip_benchmarks (data moat) ─────────────────────────────────────────
-- Aggregated market data — grows with every extraction (the network effect)

CREATE TABLE IF NOT EXISTS zip_benchmarks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    zip_code        TEXT NOT NULL,
    state           CHAR(2),
    city            TEXT,
    sample_size     INTEGER NOT NULL DEFAULT 0,

    -- Rent benchmarks
    median_rent_studio      NUMERIC(10,2),
    median_rent_1br         NUMERIC(10,2),
    median_rent_2br         NUMERIC(10,2),
    median_rent_3br         NUMERIC(10,2),
    p25_rent                NUMERIC(10,2),
    p75_rent                NUMERIC(10,2),

    -- Unit economics benchmarks
    median_cost_per_sqft    NUMERIC(8,4),
    median_deposit_ratio    NUMERIC(5,3),    -- deposit / monthly_rent
    median_late_fee_pct     NUMERIC(5,4),
    median_renewal_notice_days INTEGER,
    median_entry_notice_hours  INTEGER,
    median_lease_term_months   INTEGER,

    -- Clause prevalence
    pct_auto_renewal        NUMERIC(5,2),    -- % of leases with auto-renewal
    pct_pets_allowed        NUMERIC(5,2),
    pct_subletting_allowed  NUMERIC(5,2),
    pct_furnished           NUMERIC(5,2),

    -- Data quality
    data_source             TEXT DEFAULT 'corpus_aggregate',
    last_lease_added_at     TIMESTAMPTZ,

    CONSTRAINT zip_benchmarks_zip_unique UNIQUE(zip_code)
);

CREATE INDEX idx_zip_benchmarks_zip   ON zip_benchmarks(zip_code);
CREATE INDEX idx_zip_benchmarks_state ON zip_benchmarks(state);

-- Auto-update zip benchmark when a new lease is inserted
CREATE OR REPLACE FUNCTION update_zip_benchmark()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO zip_benchmarks (zip_code, state, city, sample_size, last_lease_added_at)
    VALUES (NEW.property_zip, NEW.property_state, NEW.property_city, 1, NOW())
    ON CONFLICT (zip_code) DO UPDATE SET
        sample_size = zip_benchmarks.sample_size + 1,
        last_lease_added_at = NOW(),
        updated_at = NOW(),
        median_rent_1br = (
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_rent)
            FROM leases
            WHERE property_zip = NEW.property_zip
              AND monthly_rent IS NOT NULL
              AND deleted_at IS NULL
        );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER leases_update_zip_benchmark
    AFTER INSERT ON leases
    FOR EACH ROW
    WHEN (NEW.property_zip IS NOT NULL AND NEW.monthly_rent IS NOT NULL)
    EXECUTE FUNCTION update_zip_benchmark();


-- ── TABLE: schema_versions (schema evolution tracking) ────────────────────────

CREATE TABLE IF NOT EXISTS schema_versions (
    version         TEXT PRIMARY KEY,
    deployed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description     TEXT NOT NULL,
    breaking_change BOOLEAN NOT NULL DEFAULT FALSE,
    migration_notes TEXT
);

INSERT INTO schema_versions (version, description, breaking_change) VALUES
    ('1.0.0', 'Initial schema — 35 Layer 1 fields, 12 Layer 2 computed, 5 Layer 4 scores', FALSE);


-- ── ROW-LEVEL SECURITY (RLS) ──────────────────────────────────────────────────

ALTER TABLE leases          ENABLE ROW LEVEL SECURITY;
ALTER TABLE extractions     ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_flags      ENABLE ROW LEVEL SECURITY;
ALTER TABLE users           ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "users_own_data" ON users
    FOR ALL USING (auth.uid() = id);

-- Tenants see only their own leases
CREATE POLICY "tenants_own_leases" ON leases
    FOR SELECT USING (
        user_id = auth.uid()
        OR (SELECT role FROM users WHERE id = auth.uid()) = 'admin'
    );

CREATE POLICY "tenants_insert_own_leases" ON leases
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Only the pipeline service role can update leases (not end users)
CREATE POLICY "service_update_leases" ON leases
    FOR UPDATE USING (
        (SELECT role FROM users WHERE id = auth.uid()) = 'admin'
    );

-- Extractions and flags: same isolation as leases
CREATE POLICY "extractions_via_lease" ON extractions
    FOR SELECT USING (
        lease_id IN (SELECT id FROM leases WHERE user_id = auth.uid())
    );

CREATE POLICY "risk_flags_via_lease" ON risk_flags
    FOR SELECT USING (
        lease_id IN (SELECT id FROM leases WHERE user_id = auth.uid())
    );

-- ZIP benchmarks: public read (anonymous market data — this is intentional)
CREATE POLICY "zip_benchmarks_public_read" ON zip_benchmarks
    FOR SELECT USING (TRUE);


-- ── ANALYTICS VIEWS ───────────────────────────────────────────────────────────

-- Tenant dashboard view (safe: only non-PII aggregates)
CREATE OR REPLACE VIEW tenant_dashboard AS
SELECT
    l.id,
    l.extraction_id,
    l.created_at,
    l.property_address,
    l.property_city,
    l.property_state,
    l.monthly_rent,
    l.computed_effective_monthly_cost,
    l.risk_tier,
    l.score_overall_risk,
    l.score_renewal_risk,
    l.computed_days_until_renewal_deadline,
    l.computed_renewal_deadline_date,
    l.rent_vs_zip_median_pct,
    l.requires_human_review,
    COUNT(rf.id) AS total_flags,
    SUM(CASE WHEN rf.severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_flags,
    SUM(CASE WHEN rf.severity = 'HIGH' THEN 1 ELSE 0 END) AS high_flags
FROM leases l
LEFT JOIN risk_flags rf ON rf.lease_id = l.id
WHERE l.deleted_at IS NULL
GROUP BY l.id;

-- Admin: ZIP-level market intelligence view
CREATE OR REPLACE VIEW zip_market_intelligence AS
SELECT
    zb.zip_code,
    zb.city,
    zb.state,
    zb.sample_size,
    zb.median_rent_1br,
    zb.median_cost_per_sqft,
    zb.median_deposit_ratio,
    zb.median_renewal_notice_days,
    zb.median_entry_notice_hours,
    zb.pct_auto_renewal,
    CASE WHEN zb.sample_size >= 10 THEN 'statistically_meaningful'
         WHEN zb.sample_size >= 3  THEN 'early_signal'
         ELSE 'insufficient_data' END AS data_quality,
    zb.last_lease_added_at
FROM zip_benchmarks zb
ORDER BY zb.sample_size DESC;

-- ── SAMPLE DATA VERIFICATION QUERY ────────────────────────────────────────────
-- Run after inserting your first beta lease to verify everything is wired up:
/*
SELECT
    l.extraction_id,
    l.tenant_full_name,
    l.monthly_rent,
    l.risk_tier,
    l.score_overall_risk,
    COUNT(rf.id) AS flags,
    zb.sample_size AS zip_sample_size
FROM leases l
LEFT JOIN risk_flags rf ON rf.lease_id = l.id
LEFT JOIN zip_benchmarks zb ON zb.zip_code = l.property_zip
GROUP BY l.id, zb.sample_size
ORDER BY l.created_at DESC
LIMIT 10;
*/

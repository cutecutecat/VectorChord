-- List of shell types

CREATE TYPE scalar8;
CREATE TYPE sphere_vector;
CREATE TYPE sphere_halfvec;
CREATE TYPE sphere_scalar8;

-- List of data types

CREATE TYPE scalar8 (
    INPUT = _vchord_scalar8_in,
    OUTPUT = _vchord_scalar8_out,
    RECEIVE = _vchord_scalar8_recv,
    SEND = _vchord_scalar8_send,
    TYPMOD_IN = _vchord_typmod_in_65535,
    TYPMOD_OUT = _vchord_typmod_out,
    STORAGE = EXTERNAL,
    INTERNALLENGTH = VARIABLE,
    ALIGNMENT = double
);

CREATE TYPE sphere_vector AS (
    center vector,
    radius REAL
);

CREATE TYPE sphere_halfvec AS (
    center halfvec,
    radius REAL
);

CREATE TYPE sphere_scalar8 AS (
    center scalar8,
    radius REAL
);

-- List of operators

CREATE OPERATOR <-> (
    PROCEDURE = _vchord_scalar8_operator_l2,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <->
);

CREATE OPERATOR <#> (
    PROCEDURE = _vchord_scalar8_operator_ip,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <#>
);

CREATE OPERATOR <=> (
    PROCEDURE = _vchord_scalar8_operator_cosine,
    LEFTARG = scalar8,
    RIGHTARG = scalar8,
    COMMUTATOR = <=>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_vector_sphere_l2_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_halfvec_sphere_l2_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<->> (
    PROCEDURE = _vchord_scalar8_sphere_l2_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<->>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_vector_sphere_ip_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_halfvec_sphere_ip_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<#>> (
    PROCEDURE = _vchord_scalar8_sphere_ip_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<#>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_vector_sphere_cosine_in,
    LEFTARG = vector,
    RIGHTARG = sphere_vector,
    COMMUTATOR = <<=>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_halfvec_sphere_cosine_in,
    LEFTARG = halfvec,
    RIGHTARG = sphere_halfvec,
    COMMUTATOR = <<=>>
);

CREATE OPERATOR <<=>> (
    PROCEDURE = _vchord_scalar8_sphere_cosine_in,
    LEFTARG = scalar8,
    RIGHTARG = sphere_scalar8,
    COMMUTATOR = <<=>>
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_vector_operator_maxsim,
    LEFTARG = vector[],
    RIGHTARG = vector[]
);

CREATE OPERATOR @# (
    PROCEDURE = _vchord_halfvec_operator_maxsim,
    LEFTARG = halfvec[],
    RIGHTARG = halfvec[]
);

-- List of functions

CREATE FUNCTION sphere(vector, real) RETURNS sphere_vector
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION sphere(halfvec, real) RETURNS sphere_halfvec
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION sphere(scalar8, real) RETURNS sphere_scalar8
IMMUTABLE PARALLEL SAFE LANGUAGE sql AS 'SELECT ROW($1, $2)';

CREATE FUNCTION quantize_to_scalar8(vector) RETURNS scalar8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_vector_quantize_to_scalar8_wrapper';

CREATE FUNCTION quantize_to_scalar8(halfvec) RETURNS scalar8
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchord_halfvec_quantize_to_scalar8_wrapper';

CREATE FUNCTION vchordrq_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_amhandler_wrapper';

CREATE FUNCTION vchordrq_prewarm(regclass, integer default 0) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordrq_prewarm_wrapper';

CREATE FUNCTION vchordg_amhandler(internal) RETURNS index_am_handler
IMMUTABLE STRICT PARALLEL SAFE LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_amhandler_wrapper';

CREATE FUNCTION vchordg_prewarm(regclass) RETURNS TEXT
STRICT LANGUAGE c AS 'MODULE_PATHNAME', '_vchordg_prewarm_wrapper';

CREATE FUNCTION vchordrq_evaluate_query_recall(
    query text,
    exact_search boolean default False,
    gt_epsilon real default 1.9
)
RETURNS real
LANGUAGE plpgsql
STRICT
AS $$
DECLARE
    rough tid[];
    accu tid[];
    match_count integer := 0;
    gt_count integer;
    recall real;
    iter_ctid tid;
    top_k integer;
    query_probes text;
    gt_probes text;

BEGIN
    IF NOT exact_search THEN
        BEGIN
            query_probes := current_setting('vchordrq.probes');
        END;
    END IF;

    BEGIN
        EXECUTE
            format('SELECT array_agg(id) FROM (%s) AS result(id)', query)
        INTO
            rough;
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Error executing ANN query "%": %', query, SQLERRM;
    END;

    top_k := cardinality(rough);
    IF top_k IS NULL OR top_k = 0 THEN
        RAISE WARNING 'ANN query returned 0 results. Please check and run the query.';
        RETURN 0.0;
    END IF;

    BEGIN
        IF exact_search THEN
            SET LOCAL vchordrq.enable_scan = off;
        ELSE
            IF query_probes = '' THEN
                gt_probes := '';
            ELSIF position(',' in query_probes) > 0 THEN
                gt_probes := '9999,9999';
            ELSE
                gt_probes := '9999';
            END IF;
            EXECUTE format('SET LOCAL "vchordrq.probes" = %L', gt_probes);
            EXECUTE format('SET LOCAL "vchordrq.epsilon" = %L', gt_epsilon);
            RESET vchordrq.max_scan_tuples;
        END IF;
        EXECUTE
            format('SELECT array_agg(id) FROM (%s) AS result(id)', query)
        INTO
            accu;
    EXCEPTION WHEN OTHERS THEN
         RAISE EXCEPTION 'Error executing Ground Truth query "%": %', query, SQLERRM;
    END;

    gt_count := cardinality(accu);
    IF gt_count IS NULL OR gt_count = 0 THEN
        RAISE WARNING 'Ground Truth query returned 0 results. Please check and run the query.';
        RETURN 0.0;
    END IF;
    IF gt_count != top_k THEN
         RAISE NOTICE 'Ground Truth query returned % results, while ANN query returned % results.', gt_count, top_k;
    END IF;

    SELECT COUNT(*) INTO match_count FROM (SELECT unnest(rough) INTERSECT SELECT unnest(accu)) AS tids;
    recall := match_count::real / gt_count::real;
    RETURN recall;
END;
$$;

-- List of access methods

CREATE ACCESS METHOD vchordrq TYPE INDEX HANDLER vchordrq_amhandler;
CREATE ACCESS METHOD vchordg TYPE INDEX HANDLER vchordg_amhandler;

-- List of operator families

CREATE OPERATOR FAMILY vector_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY halfvec_maxsim_ops USING vchordrq;
CREATE OPERATOR FAMILY vector_l2_ops USING vchordg;
CREATE OPERATOR FAMILY vector_ip_ops USING vchordg;
CREATE OPERATOR FAMILY vector_cosine_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_l2_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_ip_ops USING vchordg;
CREATE OPERATOR FAMILY halfvec_cosine_ops USING vchordg;

-- List of operator classes

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordrq FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordrq FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordrq FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordrq FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordrq_support_halfvec_cosine_ops();

CREATE OPERATOR CLASS vector_maxsim_ops
    FOR TYPE vector[] USING vchordrq FAMILY vector_maxsim_ops AS
    OPERATOR 3 @# (vector[], vector[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_vector_maxsim_ops();

CREATE OPERATOR CLASS halfvec_maxsim_ops
    FOR TYPE halfvec[] USING vchordrq FAMILY halfvec_maxsim_ops AS
    OPERATOR 3 @# (halfvec[], halfvec[]) FOR ORDER BY float_ops,
    FUNCTION 1 _vchordrq_support_halfvec_maxsim_ops();

CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING vchordg FAMILY vector_l2_ops AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_l2_ops();

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING vchordg FAMILY vector_ip_ops AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_ip_ops();

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING vchordg FAMILY vector_cosine_ops AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (vector, sphere_vector) FOR SEARCH,
    FUNCTION 1 _vchordg_support_vector_cosine_ops();

CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_l2_ops AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<->> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_l2_ops();

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_ip_ops AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<#>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_ip_ops();

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING vchordg FAMILY halfvec_cosine_ops AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    OPERATOR 2 <<=>> (halfvec, sphere_halfvec) FOR SEARCH,
    FUNCTION 1 _vchordg_support_halfvec_cosine_ops();

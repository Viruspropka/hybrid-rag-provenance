from hybrid_rag.notes import chunk_note, parse_frontmatter, verified_epoch


def test_frontmatter_fields_and_quotes():
    fm, body = parse_frontmatter(
        '---\nid: x1\ntitle: "ADR-1: A: B"\ntype: Runbook\nstatus: Superseded\ntier: 2\n'
        "reliability: 0.75\ngenerated_by: Model:Foo\nverified: 2026-01-02\nvolatility: FAST\n"
        "tags: [a, 'b', c]\n---\n# T\n")
    assert fm.id == "x1" and fm.title == "ADR-1: A: B" and fm.doc_type == "runbook"
    assert fm.status == "superseded" and fm.tier == 2 and fm.reliability == 0.75
    assert fm.generated_by == "model:foo" and fm.verified == "2026-01-02" and fm.volatility == "fast"
    assert fm.tags == ["a", "b", "c"]
    assert body == "# T\n"


def test_bad_values_fall_back_to_defaults():
    fm, _ = parse_frontmatter("---\ntier: high\nreliability: 7\nverified: someday\n---\nbody")
    assert fm.tier is None and fm.reliability is None
    assert verified_epoch(fm.verified) is None


def test_no_frontmatter_keeps_text():
    fm, body = parse_frontmatter("# just text\n")
    assert fm.status == "active" and body == "# just text\n"


def test_verified_epoch_is_utc_midnight():
    assert verified_epoch("1970-01-02") == 86400.0


def test_chunks_follow_level_two_headings():
    text = "---\nid: n1\n---\n# Title\n\n## One\n" + "alpha " * 40 + "\n\n## Two\n" + "beta " * 40
    chunks = chunk_note("x/n1.md", text)
    assert [c.section_heading for c in chunks] == ["One", "Two"]
    assert all(c.note_id == "n1" and c.title == "Title" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 2


def test_long_section_is_windowed_with_overlap():
    text = "# T\n\n## Long\n" + " ".join(f"w{i}" for i in range(3000))
    chunks = chunk_note("long.md", text, max_chars=1000, overlap=200)
    assert len(chunks) > 3
    assert all(len(c.text) <= 1000 + 120 for c in chunks)   # a short title piece may be merged in
    assert chunks[0].text[-150:] in chunks[0].text and chunks[1].text[:150] in text


def test_short_sections_are_merged_not_dropped():
    text = "---\nid: n\n---\n# Title\n\n## A\nshort a\n\n## B\nshort b\n"
    chunks = chunk_note("n.md", text)
    assert len(chunks) == 1
    assert "short a" in chunks[0].text and "short b" in chunks[0].text and "# Title" in chunks[0].text


def test_sensitive_note_is_skipped_and_stub_kept():
    assert chunk_note("s.md", "---\nsensitive: true\n---\n# Secret\n\n## A\n" + "x " * 200) == []
    stub = chunk_note("folder/stub.md", "# Stub\n\ntiny")
    assert len(stub) == 1 and stub[0].note_id == "stub"


def test_provenance_is_copied_to_every_chunk():
    text = "---\nid: n\nreliability: 0.6\nverified: 2026-02-03\ntier: 3\n---\n## A\n" + "a " * 100 + "\n## B\n" + "b " * 100
    for c in chunk_note("n.md", text):
        assert c.reliability == 0.6 and c.tier == 3 and c.verified == verified_epoch("2026-02-03")

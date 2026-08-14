"""Unit tests for generic subject normalization."""
from __future__ import annotations

from mailagent.domain.models import NormalizedSubject
from mailagent.preprocessing.subject_normalizer import normalize_subject


class TestPrefixStripping:
    """Reply / forward prefix stripping scenarios."""

    def test_single_re_prefix_stripped(self) -> None:
        """A single 'Re:' prefix is stripped from the subject."""
        ns = normalize_subject("Re: Status Report")
        assert ns.clean == "Status Report"

    def test_single_fwd_prefix_stripped(self) -> None:
        """A single 'Fwd:' prefix is stripped (case-insensitive)."""
        ns = normalize_subject("Fwd: Schedule Update")
        assert ns.clean == "Schedule Update"

    def test_chinese_reply_prefix_stripped(self) -> None:
        """Chinese '回复:' prefix is stripped."""
        ns = normalize_subject("回复: 船期更新")
        assert ns.clean == "船期更新"

    def test_chinese_forward_prefix_stripped(self) -> None:
        """Chinese '转发:' prefix is stripped."""
        ns = normalize_subject("转发: 靠泊申请")
        assert ns.clean == "靠泊申请"

    def test_bracket_re_prefix_stripped(self) -> None:
        """'[Re]' bracket prefix is stripped."""
        ns = normalize_subject("[Re] Status Report")
        assert ns.clean == "Status Report"

    def test_nested_prefixes_stripped_iteratively(self) -> None:
        """Nested prefixes 'Re[3]: Re: 回复: ...' are stripped iteratively."""
        ns = normalize_subject("Re[3]: Re: 回复: Status Report")
        assert ns.clean == "Status Report"

    def test_re_numbered_prefix_stripped(self) -> None:
        """'Re[2]:' numbered prefix is stripped."""
        ns = normalize_subject("Re[2]: Arrival Notice")
        assert ns.clean == "Arrival Notice"

    def test_external_email_marker_stripped(self) -> None:
        """'【外部邮件】' marker is stripped."""
        ns = normalize_subject("【外部邮件】Re: Test Subject")
        assert ns.clean == "Test Subject"

    def test_external_email_marker_alone_stripped(self) -> None:
        """'【外部邮件】' alone (without reply prefix) is stripped."""
        ns = normalize_subject("【外部邮件】Important Notice")
        assert ns.clean == "Important Notice"

    def test_no_prefix_returns_as_is(self) -> None:
        """Subject without any prefix is returned as-is (after whitespace folding)."""
        ns = normalize_subject("Status Report")
        assert ns.clean == "Status Report"

    def test_prefix_not_inside_word(self) -> None:
        """'re' inside a word (e.g. 'are:') is not stripped."""
        ns = normalize_subject("are: something")
        assert ns.clean == "are: something"


class TestWhitespaceFolding:
    """NBSP and consecutive whitespace folding scenarios."""

    def test_nbsp_folded_to_space(self) -> None:
        """NBSP (\\xa0) is folded into a regular space."""
        ns = normalize_subject("STATUS\u00a0CHANGE EXAMPLE")
        assert ns.clean == "STATUS CHANGE EXAMPLE"

    def test_multiple_spaces_collapsed(self) -> None:
        """Multiple consecutive spaces are collapsed into one."""
        ns = normalize_subject("STATUS CHANGE   EXAMPLE")
        assert ns.clean == "STATUS CHANGE EXAMPLE"

    def test_nbsp_and_multiple_spaces_folded(self) -> None:
        """NBSP and multiple spaces are folded together."""
        ns = normalize_subject("STATUS\u00a0CHANGE   EXAMPLE")
        assert ns.clean == "STATUS CHANGE EXAMPLE"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        """Leading and trailing whitespace is stripped."""
        ns = normalize_subject("  Status Report  ")
        assert ns.clean == "Status Report"


class TestNormalizedSubjectFields:
    """NormalizedSubject remains generic and contains no vertical fields."""

    def test_raw_field_preserved(self) -> None:
        """The 'raw' field preserves the original input string."""
        raw = "Re[3]: Re: 回复: Status Report"
        ns = normalize_subject(raw)
        assert ns.raw == raw

    def test_model_contains_only_generic_subject_fields(self) -> None:
        ns = normalize_subject("Berlin Example STATUS CHANGE")
        assert ns.model_dump() == {
            "raw": "Berlin Example STATUS CHANGE",
            "clean": "Berlin Example STATUS CHANGE",
        }

    def test_returns_normalized_subject_instance(self) -> None:
        """normalize_subject returns a NormalizedSubject instance."""
        ns = normalize_subject("Test")
        assert isinstance(ns, NormalizedSubject)

    def test_empty_string(self) -> None:
        """Empty string input produces empty clean field."""
        ns = normalize_subject("")
        assert ns.raw == ""
        assert ns.clean == ""

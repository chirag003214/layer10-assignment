from __future__ import annotations

from ingestion.preprocessor import RawDocument


def bug_report_doc() -> RawDocument:
    """Bug report mentioning EKF3, Copter 4.5.7, AHRS_EKF_TYPE, and a user."""
    return RawDocument(
        doc_id="inttest-bug-001",
        doc_type="issue",
        issue_number=99001,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99001",
        author="test-pilot",
        created_at="2024-11-15T10:30:00Z",
        title="EKF3 crash on Copter 4.5.7 when AHRS_EKF_TYPE is changed",
        body=(
            "After upgrading to Copter 4.5.7, my quadcopter experienced an EKF3 "
            "failsafe and crashed during a routine flight. The issue occurs when "
            "AHRS_EKF_TYPE is set to 3. The EKF3 lane switch happens mid-flight and "
            "causes a sudden attitude estimation failure.\n\n"
            "Steps to reproduce:\n"
            "1. Flash Copter 4.5.7 firmware\n"
            "2. Set AHRS_EKF_TYPE = 3\n"
            "3. Arm and take off in Loiter mode\n"
            "4. After ~2 minutes the EKF3 primary lane switches and the copter crashes\n\n"
            "Logs attached. @tridge do you have any insight into this?"
        ),
        labels=["Bug", "EKF"],
        state="open",
    )


def fix_confirmation_doc() -> RawDocument:
    """Comment confirming a fix in Copter 4.5.8."""
    return RawDocument(
        doc_id="inttest-fix-001",
        doc_type="comment",
        issue_number=99001,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99001#comment-1",
        author="tridge",
        created_at="2024-12-01T08:00:00Z",
        title="EKF3 crash on Copter 4.5.7 when AHRS_EKF_TYPE is changed",
        body=(
            "This has been fixed in Copter 4.5.8. The root cause was a race condition "
            "in the EKF3 lane switching logic that was triggered when AHRS_EKF_TYPE "
            "was changed in flight. The fix ensures the lane switch is completed "
            "atomically before attitude estimation resumes."
        ),
        labels=["Bug", "EKF"],
    )


def duplicate_issue_doc() -> RawDocument:
    """Issue explicitly marked as a duplicate."""
    return RawDocument(
        doc_id="inttest-dup-001",
        doc_type="issue",
        issue_number=99002,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99002",
        author="copter-user42",
        created_at="2024-11-20T14:00:00Z",
        title="EKF3 failsafe after firmware update - duplicate",
        body=(
            "I'm seeing EKF3 failsafe errors on Copter 4.5.7 after upgrading. "
            "The EKF3 attitude estimation fails during flight and triggers a crash. "
            "This is a duplicate of issue #99001 which reports the same EKF3 "
            "crash on Copter 4.5.7. Closing this as duplicate of #99001."
        ),
        labels=["Bug", "Duplicate"],
        state="closed",
        state_reason="not_planned",
    )


def parameter_discussion_doc() -> RawDocument:
    """Discussion about configuring parameters for a component."""
    return RawDocument(
        doc_id="inttest-param-001",
        doc_type="issue",
        issue_number=99003,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99003",
        author="rover-fan",
        created_at="2024-10-05T09:15:00Z",
        title="Recommended GPS_TYPE setting for Here3 GPS module",
        body=(
            "I'm trying to configure my Here3 GPS module with ArduPilot. "
            "The GPS driver needs GPS_TYPE set to 2 for the Here3 to work correctly "
            "over CAN. You also need to set CAN_P1_DRIVER = 1 to enable the CAN bus. "
            "Without GPS_TYPE = 2, the GPS driver does not initialize the Here3 and "
            "no position fix is obtained."
        ),
        labels=["Documentation", "GPS"],
    )


def empty_body_doc() -> RawDocument:
    """Issue with no body text — should return empty extraction."""
    return RawDocument(
        doc_id="inttest-empty-001",
        doc_type="issue",
        issue_number=99004,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99004",
        author="ghost-user",
        created_at="2024-09-01T00:00:00Z",
        title="",
        body="",
        labels=[],
    )


def multi_component_doc() -> RawDocument:
    """Issue mentioning EKF2 and EKF3 as distinct components."""
    return RawDocument(
        doc_id="inttest-multi-001",
        doc_type="issue",
        issue_number=99005,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99005",
        author="nav-dev",
        created_at="2024-11-10T12:00:00Z",
        title="EKF2 vs EKF3 performance comparison on Plane",
        body=(
            "I have been comparing EKF2 and EKF3 on a fixed-wing Plane build. "
            "EKF2 converges faster on startup but EKF3 gives smoother attitude "
            "estimates in gusty conditions. When switching from EKF2 to EKF3 mid-flight "
            "via AHRS_EKF_TYPE there is a brief position glitch. Both EKF2 and EKF3 "
            "use the same IMU data but their internal state machines differ significantly."
        ),
        labels=["Enhancement", "EKF"],
    )


def fuzzy_merge_pair_docs() -> tuple[RawDocument, RawDocument]:
    """Two docs mentioning the same component with different surface forms."""
    doc_a = RawDocument(
        doc_id="inttest-fuzzy-001a",
        doc_type="issue",
        issue_number=99006,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99006",
        author="script-dev",
        created_at="2024-08-20T10:00:00Z",
        title="Scripting backend crash on Plane 4.4.0",
        body=(
            "The scripting backend crashes when loading a Lua script that exceeds "
            "the memory limit. The scripting backend should return an error instead "
            "of segfaulting. This affects the scripting backend reliability on "
            "long-running missions."
        ),
        labels=["Bug", "Scripting"],
    )
    doc_b = RawDocument(
        doc_id="inttest-fuzzy-001b",
        doc_type="issue",
        issue_number=99007,
        source_url="https://github.com/ArduPilot/ardupilot/issues/99007",
        author="lua-user",
        created_at="2024-08-25T15:00:00Z",
        title="Lua scripting timeout on Copter",
        body=(
            "The Lua scripting engine times out when running complex navigation "
            "scripts on Copter. The Lua scripting engine should support configurable "
            "timeouts via SCR_VM_I_COUNT. The Lua scripting system needs better "
            "memory management for large scripts."
        ),
        labels=["Bug", "Scripting"],
    )
    return doc_a, doc_b
